// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package parquet

import (
	"context"
	"fmt"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/parquet/pqarrow"
	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

const (
	sparseVectorIndice = "indices"
	sparseVectorValues = "values"
)

func WrapTypeErr(expect string, actual string, field *schemapb.FieldSchema) error {
	return merr.WrapErrImportFailed(
		fmt.Sprintf("expect '%s' type for field '%s', but got '%s' type",
			expect, field.GetName(), actual))
}

func CreateFieldReaders(ctx context.Context, fileReader *pqarrow.FileReader, schema *schemapb.CollectionSchema) (map[int64]*FieldReader, error) {
	nameToField := lo.KeyBy(schema.GetFields(), func(field *schemapb.FieldSchema) string {
		return field.GetName()
	})

	pqSchema, err := fileReader.Schema()
	if err != nil {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("get parquet schema failed, err=%v", err))
	}

	err = isSchemaEqual(schema, pqSchema)
	if err != nil {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("schema not equal, err=%v", err))
	}

	// this loop is for "how many fields are provided by this parquet file?"
	readFields := make(map[string]int64)
	crs := make(map[int64]*FieldReader)
	for i, pqField := range pqSchema.Fields() {
		field, ok := nameToField[pqField.Name]
		if !ok {
			// redundant fields, ignore. only accepts a special field "$meta" to store dynamic data
			continue
		}

		// auto-id field must not provided
		if typeutil.IsAutoPKField(field) {
			return nil, merr.WrapErrImportFailed(
				fmt.Sprintf("the primary key '%s' is auto-generated, no need to provide", field.GetName()))
		}
		// function output field must not provided
		if field.GetIsFunctionOutput() {
			return nil, merr.WrapErrImportFailed(
				fmt.Sprintf("the field '%s' is output by function, no need to provide", field.GetName()))
		}

		cr, err := NewFieldReader(ctx, fileReader, i, field)
		if err != nil {
			return nil, err
		}
		if _, ok = crs[field.GetFieldID()]; ok {
			return nil, merr.WrapErrImportFailed(
				fmt.Sprintf("there is multi field with name: %s", field.GetName()))
		}
		crs[field.GetFieldID()] = cr
		readFields[field.GetName()] = field.GetFieldID()
	}

	// this loop is for "are there any fields not provided in the parquet file?"
	for _, field := range nameToField {
		// auto-id field, function output field already checked
		// dynamic field, nullable field, default value field, not provided or provided both ok
		if typeutil.IsAutoPKField(field) || field.GetIsDynamic() || field.GetIsFunctionOutput() ||
			field.GetNullable() || field.GetDefaultValue() != nil {
			continue
		}
		// the other field must be provided
		if _, ok := crs[field.GetFieldID()]; !ok {
			return nil, merr.WrapErrImportFailed(
				fmt.Sprintf("no parquet field for milvus field '%s'", field.GetName()))
		}
	}

	log.Info("create parquet column readers", zap.Any("readFields", readFields))
	return crs, nil
}

func isArrowIntegerType(dataType arrow.Type) bool {
	switch dataType {
	case arrow.INT8, arrow.INT16, arrow.INT32, arrow.INT64:
		return true
	default:
		return false
	}
}

func isArrowFloatingType(dataType arrow.Type) bool {
	switch dataType {
	case arrow.FLOAT32, arrow.FLOAT64:
		return true
	default:
		return false
	}
}

func isArrowArithmeticType(dataType arrow.Type) bool {
	return isArrowIntegerType(dataType) || isArrowFloatingType(dataType)
}

func isArrowDataTypeConvertible(src arrow.DataType, dst arrow.DataType, field *schemapb.FieldSchema) bool {
	srcType := src.ID()
	dstType := dst.ID()
	switch srcType {
	case arrow.BOOL:
		return dstType == arrow.BOOL
	case arrow.UINT8:
		return dstType == arrow.UINT8
	case arrow.INT8:
		return isArrowArithmeticType(dstType)
	case arrow.INT16:
		return isArrowArithmeticType(dstType) && dstType != arrow.INT8
	case arrow.INT32:
		return isArrowArithmeticType(dstType) && dstType != arrow.INT8 && dstType != arrow.INT16
	case arrow.INT64:
		return isArrowFloatingType(dstType) || dstType == arrow.INT64
	case arrow.FLOAT32:
		return isArrowFloatingType(dstType)
	case arrow.FLOAT64:
		// TODO caiyd: need do strict type check
		// return dstType == arrow.FLOAT64
		return isArrowFloatingType(dstType)
	case arrow.STRING:
		return dstType == arrow.STRING
	case arrow.BINARY:
		return dstType == arrow.LIST && dst.(*arrow.ListType).Elem().ID() == arrow.UINT8
	case arrow.LIST:
		return dstType == arrow.LIST && isArrowDataTypeConvertible(src.(*arrow.ListType).Elem(), dst.(*arrow.ListType).Elem(), field)
	case arrow.NULL:
		// if nullable==true or has set default_value, can use null type
		return field.GetNullable() || field.GetDefaultValue() != nil
	case arrow.STRUCT:
		if field.GetDataType() == schemapb.DataType_SparseFloatVector {
			valid, _ := IsValidSparseVectorSchema(src)
			return valid
		}
		return false
	default:
		return false
	}
}

// This method returns two booleans
// The first boolean value means the arrowType is a valid sparse vector schema
// The second boolean value: true means the sparse vector is stored as JSON-format string,
// false means the sparse vector is stored as parquet struct
func IsValidSparseVectorSchema(arrowType arrow.DataType) (bool, bool) {
	arrowID := arrowType.ID()
	if arrowID == arrow.STRUCT {
		arrType := arrowType.(*arrow.StructType)
		indicesType, ok1 := arrType.FieldByName(sparseVectorIndice)
		valuesType, ok2 := arrType.FieldByName(sparseVectorValues)
		if !ok1 || !ok2 {
			return false, false
		}

		// indices can be uint32 list or int64 list
		// values can be float32 list or float64 list
		isValidType := func(finger string, expectedType arrow.DataType) bool {
			return finger == arrow.ListOf(expectedType).Fingerprint()
		}
		indicesFinger := indicesType.Type.Fingerprint()
		valuesFinger := valuesType.Type.Fingerprint()
		indicesTypeIsOK := (isValidType(indicesFinger, arrow.PrimitiveTypes.Int32) ||
			isValidType(indicesFinger, arrow.PrimitiveTypes.Uint32) ||
			isValidType(indicesFinger, arrow.PrimitiveTypes.Int64) ||
			isValidType(indicesFinger, arrow.PrimitiveTypes.Uint64))
		valuesTypeIsOK := (isValidType(valuesFinger, arrow.PrimitiveTypes.Float32) ||
			isValidType(valuesFinger, arrow.PrimitiveTypes.Float64))
		return indicesTypeIsOK && valuesTypeIsOK, false
	}
	return arrowID == arrow.STRING, true
}

func convertToArrowDataType(field *schemapb.FieldSchema, isArray bool) (arrow.DataType, error) {
	dataType := field.GetDataType()
	if isArray {
		dataType = field.GetElementType()
	}
	switch dataType {
	case schemapb.DataType_Bool:
		return &arrow.BooleanType{}, nil
	case schemapb.DataType_Int8:
		return &arrow.Int8Type{}, nil
	case schemapb.DataType_Int16:
		return &arrow.Int16Type{}, nil
	case schemapb.DataType_Int32:
		return &arrow.Int32Type{}, nil
	case schemapb.DataType_Int64:
		return &arrow.Int64Type{}, nil
	case schemapb.DataType_Float:
		return &arrow.Float32Type{}, nil
	case schemapb.DataType_Double:
		return &arrow.Float64Type{}, nil
	case schemapb.DataType_VarChar, schemapb.DataType_String:
		return &arrow.StringType{}, nil
	case schemapb.DataType_JSON:
		return &arrow.StringType{}, nil
	case schemapb.DataType_Array:
		elemType, err := convertToArrowDataType(field, true)
		if err != nil {
			return nil, err
		}
		return arrow.ListOfField(arrow.Field{
			Name:     "item",
			Type:     elemType,
			Nullable: true,
			Metadata: arrow.Metadata{},
		}), nil
	case schemapb.DataType_BinaryVector, schemapb.DataType_Float16Vector, schemapb.DataType_BFloat16Vector:
		return arrow.ListOfField(arrow.Field{
			Name:     "item",
			Type:     &arrow.Uint8Type{},
			Nullable: true,
			Metadata: arrow.Metadata{},
		}), nil
	case schemapb.DataType_FloatVector:
		return arrow.ListOfField(arrow.Field{
			Name:     "item",
			Type:     &arrow.Float32Type{},
			Nullable: true,
			Metadata: arrow.Metadata{},
		}), nil
	case schemapb.DataType_SparseFloatVector:
		return &arrow.StringType{}, nil
	case schemapb.DataType_Int8Vector:
		return arrow.ListOfField(arrow.Field{
			Name:     "item",
			Type:     &arrow.Int8Type{},
			Nullable: true,
			Metadata: arrow.Metadata{},
		}), nil
	default:
		return nil, merr.WrapErrParameterInvalidMsg("unsupported data type %v", dataType.String())
	}
}

// This method is used only by import util and related tests. Returned arrow.Schema
// doesn't include function output fields.
func ConvertToArrowSchema(schema *schemapb.CollectionSchema, useNullType bool) (*arrow.Schema, error) {
	arrFields := make([]arrow.Field, 0)
	for _, field := range schema.GetFields() {
		if typeutil.IsAutoPKField(field) || field.GetIsFunctionOutput() {
			continue
		}
		arrDataType, err := convertToArrowDataType(field, false)
		if err != nil {
			return nil, err
		}
		nullable := field.GetNullable()
		if field.GetNullable() && useNullType {
			arrDataType = arrow.Null
		}
		if field.GetDefaultValue() != nil && useNullType {
			arrDataType = arrow.Null
			nullable = true
		}
		arrFields = append(arrFields, arrow.Field{
			Name:     field.GetName(),
			Type:     arrDataType,
			Nullable: nullable,
			Metadata: arrow.Metadata{},
		})
	}
	return arrow.NewSchema(arrFields, nil), nil
}

func isSchemaEqual(schema *schemapb.CollectionSchema, arrSchema *arrow.Schema) error {
	arrNameToField := lo.KeyBy(arrSchema.Fields(), func(field arrow.Field) string {
		return field.Name
	})
	for _, field := range schema.GetFields() {
		// ignore autoPKField and functionOutputField
		if typeutil.IsAutoPKField(field) || field.GetIsFunctionOutput() {
			continue
		}
		arrField, ok := arrNameToField[field.GetName()]
		if !ok {
			// Special fields no need to provide in data files, the parquet file doesn't contain this field, no need to compare
			// 1. dynamic field(name is "$meta"), ignore
			// 2. nullable field, filled with null values
			// 3. default value field, filled with default value
			if field.GetIsDynamic() || field.GetNullable() || field.GetDefaultValue() != nil {
				continue
			}
			return merr.WrapErrImportFailed(fmt.Sprintf("field '%s' not in arrow schema", field.GetName()))
		}
		toArrDataType, err := convertToArrowDataType(field, false)
		if err != nil {
			return err
		}
		if !isArrowDataTypeConvertible(arrField.Type, toArrDataType, field) {
			return merr.WrapErrImportFailed(fmt.Sprintf("field '%s' type mis-match, milvus data type '%s', arrow data type get '%s'",
				field.Name, field.DataType.String(), arrField.Type.String()))
		}
	}
	return nil
}

// todo(smellthemoon): use byte to store valid_data
func bytesToValidData(length int, bytes []byte) []bool {
	bools := make([]bool, 0, length)
	if len(bytes) == 0 {
		// parquet field is "optional" or "required"
		// for "required" field, the arrow.array.NullBitmapBytes() returns an empty byte list
		// which means all the elements are valid. In this case, we simply construct an all-true bool array
		for i := 0; i < length; i++ {
			bools = append(bools, true)
		}
		return bools
	}

	// for "optional" field, the arrow.array.NullBitmapBytes() returns a non-empty byte list
	// with each bit representing the existence of an element
	for i := 0; i < length; i++ {
		bit := (bytes[uint(i)/8] & BitMask[byte(i)%8]) != 0
		bools = append(bools, bit)
	}

	return bools
}

var (
	BitMask        = [8]byte{1, 2, 4, 8, 16, 32, 64, 128}
	FlippedBitMask = [8]byte{254, 253, 251, 247, 239, 223, 191, 127}
)
