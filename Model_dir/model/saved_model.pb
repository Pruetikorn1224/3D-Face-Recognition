źú
˝
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
¸
AsString

input"T

output"
Ttype:
2		
"
	precisionint˙˙˙˙˙˙˙˙˙"

scientificbool( "
shortestbool( "
widthint˙˙˙˙˙˙˙˙˙"
fillstring 
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
ş
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 

ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.12v2.4.0-49-g85c8b2a817f8ú

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 

global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
o
input_example_tensorPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_5Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_6Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_7Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_8Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_9Const*
_output_shapes
: *
dtype0*
valueB 
d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB 
Ń
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:
*
dtype0*w
valuenBl
BChinBLeftEye1BLeftEye2BMouthBottomB	MouthLeftB
MouthRightBMouthTopBNoseB	RightEye1B	RightEye2
j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 
Ő
ParseExample/ParseExampleV2ParseExampleV2input_example_tensor!ParseExample/ParseExampleV2/names'ParseExample/ParseExampleV2/sparse_keys&ParseExample/ParseExampleV2/dense_keys'ParseExample/ParseExampleV2/ragged_keysParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5ParseExample/Const_6ParseExample/Const_7ParseExample/Const_8ParseExample/Const_9*
Tdense
2
*Ô
_output_shapesÁ
ž:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*N
dense_shapes>
<::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 
Ę
9linear/linear_model/Chin/weights/part_0/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/Chin/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ć
'linear/linear_model/Chin/weights/part_0VarHandleOp*:
_class0
.,loc:@linear/linear_model/Chin/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'linear/linear_model/Chin/weights/part_0

Hlinear/linear_model/Chin/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/Chin/weights/part_0*
_output_shapes
: 
ł
.linear/linear_model/Chin/weights/part_0/AssignAssignVariableOp'linear/linear_model/Chin/weights/part_09linear/linear_model/Chin/weights/part_0/Initializer/zeros*
dtype0
Ł
;linear/linear_model/Chin/weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/Chin/weights/part_0*
_output_shapes

:*
dtype0
Ň
=linear/linear_model/LeftEye1/weights/part_0/Initializer/zerosConst*>
_class4
20loc:@linear/linear_model/LeftEye1/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ň
+linear/linear_model/LeftEye1/weights/part_0VarHandleOp*>
_class4
20loc:@linear/linear_model/LeftEye1/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+linear/linear_model/LeftEye1/weights/part_0
§
Llinear/linear_model/LeftEye1/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp+linear/linear_model/LeftEye1/weights/part_0*
_output_shapes
: 
ż
2linear/linear_model/LeftEye1/weights/part_0/AssignAssignVariableOp+linear/linear_model/LeftEye1/weights/part_0=linear/linear_model/LeftEye1/weights/part_0/Initializer/zeros*
dtype0
Ť
?linear/linear_model/LeftEye1/weights/part_0/Read/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye1/weights/part_0*
_output_shapes

:*
dtype0
Ň
=linear/linear_model/LeftEye2/weights/part_0/Initializer/zerosConst*>
_class4
20loc:@linear/linear_model/LeftEye2/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ň
+linear/linear_model/LeftEye2/weights/part_0VarHandleOp*>
_class4
20loc:@linear/linear_model/LeftEye2/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+linear/linear_model/LeftEye2/weights/part_0
§
Llinear/linear_model/LeftEye2/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp+linear/linear_model/LeftEye2/weights/part_0*
_output_shapes
: 
ż
2linear/linear_model/LeftEye2/weights/part_0/AssignAssignVariableOp+linear/linear_model/LeftEye2/weights/part_0=linear/linear_model/LeftEye2/weights/part_0/Initializer/zeros*
dtype0
Ť
?linear/linear_model/LeftEye2/weights/part_0/Read/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye2/weights/part_0*
_output_shapes

:*
dtype0
Ř
@linear/linear_model/MouthBottom/weights/part_0/Initializer/zerosConst*A
_class7
53loc:@linear/linear_model/MouthBottom/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ű
.linear/linear_model/MouthBottom/weights/part_0VarHandleOp*A
_class7
53loc:@linear/linear_model/MouthBottom/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*?
shared_name0.linear/linear_model/MouthBottom/weights/part_0
­
Olinear/linear_model/MouthBottom/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp.linear/linear_model/MouthBottom/weights/part_0*
_output_shapes
: 
Č
5linear/linear_model/MouthBottom/weights/part_0/AssignAssignVariableOp.linear/linear_model/MouthBottom/weights/part_0@linear/linear_model/MouthBottom/weights/part_0/Initializer/zeros*
dtype0
ą
Blinear/linear_model/MouthBottom/weights/part_0/Read/ReadVariableOpReadVariableOp.linear/linear_model/MouthBottom/weights/part_0*
_output_shapes

:*
dtype0
Ô
>linear/linear_model/MouthLeft/weights/part_0/Initializer/zerosConst*?
_class5
31loc:@linear/linear_model/MouthLeft/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ő
,linear/linear_model/MouthLeft/weights/part_0VarHandleOp*?
_class5
31loc:@linear/linear_model/MouthLeft/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,linear/linear_model/MouthLeft/weights/part_0
Š
Mlinear/linear_model/MouthLeft/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/MouthLeft/weights/part_0*
_output_shapes
: 
Â
3linear/linear_model/MouthLeft/weights/part_0/AssignAssignVariableOp,linear/linear_model/MouthLeft/weights/part_0>linear/linear_model/MouthLeft/weights/part_0/Initializer/zeros*
dtype0
­
@linear/linear_model/MouthLeft/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/MouthLeft/weights/part_0*
_output_shapes

:*
dtype0
Ö
?linear/linear_model/MouthRight/weights/part_0/Initializer/zerosConst*@
_class6
42loc:@linear/linear_model/MouthRight/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ř
-linear/linear_model/MouthRight/weights/part_0VarHandleOp*@
_class6
42loc:@linear/linear_model/MouthRight/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-linear/linear_model/MouthRight/weights/part_0
Ť
Nlinear/linear_model/MouthRight/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/MouthRight/weights/part_0*
_output_shapes
: 
Ĺ
4linear/linear_model/MouthRight/weights/part_0/AssignAssignVariableOp-linear/linear_model/MouthRight/weights/part_0?linear/linear_model/MouthRight/weights/part_0/Initializer/zeros*
dtype0
Ż
Alinear/linear_model/MouthRight/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/MouthRight/weights/part_0*
_output_shapes

:*
dtype0
Ň
=linear/linear_model/MouthTop/weights/part_0/Initializer/zerosConst*>
_class4
20loc:@linear/linear_model/MouthTop/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ň
+linear/linear_model/MouthTop/weights/part_0VarHandleOp*>
_class4
20loc:@linear/linear_model/MouthTop/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+linear/linear_model/MouthTop/weights/part_0
§
Llinear/linear_model/MouthTop/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp+linear/linear_model/MouthTop/weights/part_0*
_output_shapes
: 
ż
2linear/linear_model/MouthTop/weights/part_0/AssignAssignVariableOp+linear/linear_model/MouthTop/weights/part_0=linear/linear_model/MouthTop/weights/part_0/Initializer/zeros*
dtype0
Ť
?linear/linear_model/MouthTop/weights/part_0/Read/ReadVariableOpReadVariableOp+linear/linear_model/MouthTop/weights/part_0*
_output_shapes

:*
dtype0
Ę
9linear/linear_model/Nose/weights/part_0/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/Nose/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ć
'linear/linear_model/Nose/weights/part_0VarHandleOp*:
_class0
.,loc:@linear/linear_model/Nose/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'linear/linear_model/Nose/weights/part_0

Hlinear/linear_model/Nose/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/Nose/weights/part_0*
_output_shapes
: 
ł
.linear/linear_model/Nose/weights/part_0/AssignAssignVariableOp'linear/linear_model/Nose/weights/part_09linear/linear_model/Nose/weights/part_0/Initializer/zeros*
dtype0
Ł
;linear/linear_model/Nose/weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/Nose/weights/part_0*
_output_shapes

:*
dtype0
Ô
>linear/linear_model/RightEye1/weights/part_0/Initializer/zerosConst*?
_class5
31loc:@linear/linear_model/RightEye1/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ő
,linear/linear_model/RightEye1/weights/part_0VarHandleOp*?
_class5
31loc:@linear/linear_model/RightEye1/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,linear/linear_model/RightEye1/weights/part_0
Š
Mlinear/linear_model/RightEye1/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/RightEye1/weights/part_0*
_output_shapes
: 
Â
3linear/linear_model/RightEye1/weights/part_0/AssignAssignVariableOp,linear/linear_model/RightEye1/weights/part_0>linear/linear_model/RightEye1/weights/part_0/Initializer/zeros*
dtype0
­
@linear/linear_model/RightEye1/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/RightEye1/weights/part_0*
_output_shapes

:*
dtype0
Ô
>linear/linear_model/RightEye2/weights/part_0/Initializer/zerosConst*?
_class5
31loc:@linear/linear_model/RightEye2/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
ő
,linear/linear_model/RightEye2/weights/part_0VarHandleOp*?
_class5
31loc:@linear/linear_model/RightEye2/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,linear/linear_model/RightEye2/weights/part_0
Š
Mlinear/linear_model/RightEye2/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/RightEye2/weights/part_0*
_output_shapes
: 
Â
3linear/linear_model/RightEye2/weights/part_0/AssignAssignVariableOp,linear/linear_model/RightEye2/weights/part_0>linear/linear_model/RightEye2/weights/part_0/Initializer/zeros*
dtype0
­
@linear/linear_model/RightEye2/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/RightEye2/weights/part_0*
_output_shapes

:*
dtype0
Â
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0*
valueB*    
â
'linear/linear_model/bias_weights/part_0VarHandleOp*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'linear/linear_model/bias_weights/part_0

Hlinear/linear_model/bias_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/bias_weights/part_0*
_output_shapes
: 
ł
.linear/linear_model/bias_weights/part_0/AssignAssignVariableOp'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*
dtype0

;linear/linear_model/bias_weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0

8linear/linear_model/linear_model/linear_model/Chin/ShapeShapeParseExample/ParseExampleV2*
T0*
_output_shapes
:

Flinear/linear_model/linear_model/linear_model/Chin/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Hlinear/linear_model/linear_model/linear_model/Chin/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Hlinear/linear_model/linear_model/linear_model/Chin/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ź
@linear/linear_model/linear_model/linear_model/Chin/strided_sliceStridedSlice8linear/linear_model/linear_model/linear_model/Chin/ShapeFlinear/linear_model/linear_model/linear_model/Chin/strided_slice/stackHlinear/linear_model/linear_model/linear_model/Chin/strided_slice/stack_1Hlinear/linear_model/linear_model/linear_model/Chin/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Blinear/linear_model/linear_model/linear_model/Chin/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ü
@linear/linear_model/linear_model/linear_model/Chin/Reshape/shapePack@linear/linear_model/linear_model/linear_model/Chin/strided_sliceBlinear/linear_model/linear_model/linear_model/Chin/Reshape/shape/1*
N*
T0*
_output_shapes
:
Ö
:linear/linear_model/linear_model/linear_model/Chin/ReshapeReshapeParseExample/ParseExampleV2@linear/linear_model/linear_model/linear_model/Chin/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/linear/linear_model/Chin/weights/ReadVariableOpReadVariableOp'linear/linear_model/Chin/weights/part_0*
_output_shapes

:*
dtype0

 linear/linear_model/Chin/weightsIdentity/linear/linear_model/Chin/weights/ReadVariableOp*
T0*
_output_shapes

:
Ů
?linear/linear_model/linear_model/linear_model/Chin/weighted_sumMatMul:linear/linear_model/linear_model/linear_model/Chin/Reshape linear/linear_model/Chin/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

<linear/linear_model/linear_model/linear_model/LeftEye1/ShapeShapeParseExample/ParseExampleV2:1*
T0*
_output_shapes
:

Jlinear/linear_model/linear_model/linear_model/LeftEye1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Llinear/linear_model/linear_model/linear_model/LeftEye1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Llinear/linear_model/linear_model/linear_model/LeftEye1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ŕ
Dlinear/linear_model/linear_model/linear_model/LeftEye1/strided_sliceStridedSlice<linear/linear_model/linear_model/linear_model/LeftEye1/ShapeJlinear/linear_model/linear_model/linear_model/LeftEye1/strided_slice/stackLlinear/linear_model/linear_model/linear_model/LeftEye1/strided_slice/stack_1Llinear/linear_model/linear_model/linear_model/LeftEye1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Flinear/linear_model/linear_model/linear_model/LeftEye1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Dlinear/linear_model/linear_model/linear_model/LeftEye1/Reshape/shapePackDlinear/linear_model/linear_model/linear_model/LeftEye1/strided_sliceFlinear/linear_model/linear_model/linear_model/LeftEye1/Reshape/shape/1*
N*
T0*
_output_shapes
:
ŕ
>linear/linear_model/linear_model/linear_model/LeftEye1/ReshapeReshapeParseExample/ParseExampleV2:1Dlinear/linear_model/linear_model/linear_model/LeftEye1/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

3linear/linear_model/LeftEye1/weights/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye1/weights/part_0*
_output_shapes

:*
dtype0

$linear/linear_model/LeftEye1/weightsIdentity3linear/linear_model/LeftEye1/weights/ReadVariableOp*
T0*
_output_shapes

:
ĺ
Clinear/linear_model/linear_model/linear_model/LeftEye1/weighted_sumMatMul>linear/linear_model/linear_model/linear_model/LeftEye1/Reshape$linear/linear_model/LeftEye1/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

<linear/linear_model/linear_model/linear_model/LeftEye2/ShapeShapeParseExample/ParseExampleV2:2*
T0*
_output_shapes
:

Jlinear/linear_model/linear_model/linear_model/LeftEye2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Llinear/linear_model/linear_model/linear_model/LeftEye2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Llinear/linear_model/linear_model/linear_model/LeftEye2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ŕ
Dlinear/linear_model/linear_model/linear_model/LeftEye2/strided_sliceStridedSlice<linear/linear_model/linear_model/linear_model/LeftEye2/ShapeJlinear/linear_model/linear_model/linear_model/LeftEye2/strided_slice/stackLlinear/linear_model/linear_model/linear_model/LeftEye2/strided_slice/stack_1Llinear/linear_model/linear_model/linear_model/LeftEye2/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Flinear/linear_model/linear_model/linear_model/LeftEye2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Dlinear/linear_model/linear_model/linear_model/LeftEye2/Reshape/shapePackDlinear/linear_model/linear_model/linear_model/LeftEye2/strided_sliceFlinear/linear_model/linear_model/linear_model/LeftEye2/Reshape/shape/1*
N*
T0*
_output_shapes
:
ŕ
>linear/linear_model/linear_model/linear_model/LeftEye2/ReshapeReshapeParseExample/ParseExampleV2:2Dlinear/linear_model/linear_model/linear_model/LeftEye2/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

3linear/linear_model/LeftEye2/weights/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye2/weights/part_0*
_output_shapes

:*
dtype0

$linear/linear_model/LeftEye2/weightsIdentity3linear/linear_model/LeftEye2/weights/ReadVariableOp*
T0*
_output_shapes

:
ĺ
Clinear/linear_model/linear_model/linear_model/LeftEye2/weighted_sumMatMul>linear/linear_model/linear_model/linear_model/LeftEye2/Reshape$linear/linear_model/LeftEye2/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

?linear/linear_model/linear_model/linear_model/MouthBottom/ShapeShapeParseExample/ParseExampleV2:3*
T0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/MouthBottom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Olinear/linear_model/linear_model/linear_model/MouthBottom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Olinear/linear_model/linear_model/linear_model/MouthBottom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ď
Glinear/linear_model/linear_model/linear_model/MouthBottom/strided_sliceStridedSlice?linear/linear_model/linear_model/linear_model/MouthBottom/ShapeMlinear/linear_model/linear_model/linear_model/MouthBottom/strided_slice/stackOlinear/linear_model/linear_model/linear_model/MouthBottom/strided_slice/stack_1Olinear/linear_model/linear_model/linear_model/MouthBottom/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ilinear/linear_model/linear_model/linear_model/MouthBottom/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Glinear/linear_model/linear_model/linear_model/MouthBottom/Reshape/shapePackGlinear/linear_model/linear_model/linear_model/MouthBottom/strided_sliceIlinear/linear_model/linear_model/linear_model/MouthBottom/Reshape/shape/1*
N*
T0*
_output_shapes
:
ć
Alinear/linear_model/linear_model/linear_model/MouthBottom/ReshapeReshapeParseExample/ParseExampleV2:3Glinear/linear_model/linear_model/linear_model/MouthBottom/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
6linear/linear_model/MouthBottom/weights/ReadVariableOpReadVariableOp.linear/linear_model/MouthBottom/weights/part_0*
_output_shapes

:*
dtype0

'linear/linear_model/MouthBottom/weightsIdentity6linear/linear_model/MouthBottom/weights/ReadVariableOp*
T0*
_output_shapes

:
î
Flinear/linear_model/linear_model/linear_model/MouthBottom/weighted_sumMatMulAlinear/linear_model/linear_model/linear_model/MouthBottom/Reshape'linear/linear_model/MouthBottom/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=linear/linear_model/linear_model/linear_model/MouthLeft/ShapeShapeParseExample/ParseExampleV2:4*
T0*
_output_shapes
:

Klinear/linear_model/linear_model/linear_model/MouthLeft/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Mlinear/linear_model/linear_model/linear_model/MouthLeft/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Mlinear/linear_model/linear_model/linear_model/MouthLeft/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ĺ
Elinear/linear_model/linear_model/linear_model/MouthLeft/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/MouthLeft/ShapeKlinear/linear_model/linear_model/linear_model/MouthLeft/strided_slice/stackMlinear/linear_model/linear_model/linear_model/MouthLeft/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/MouthLeft/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Glinear/linear_model/linear_model/linear_model/MouthLeft/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Elinear/linear_model/linear_model/linear_model/MouthLeft/Reshape/shapePackElinear/linear_model/linear_model/linear_model/MouthLeft/strided_sliceGlinear/linear_model/linear_model/linear_model/MouthLeft/Reshape/shape/1*
N*
T0*
_output_shapes
:
â
?linear/linear_model/linear_model/linear_model/MouthLeft/ReshapeReshapeParseExample/ParseExampleV2:4Elinear/linear_model/linear_model/linear_model/MouthLeft/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
4linear/linear_model/MouthLeft/weights/ReadVariableOpReadVariableOp,linear/linear_model/MouthLeft/weights/part_0*
_output_shapes

:*
dtype0

%linear/linear_model/MouthLeft/weightsIdentity4linear/linear_model/MouthLeft/weights/ReadVariableOp*
T0*
_output_shapes

:
č
Dlinear/linear_model/linear_model/linear_model/MouthLeft/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/MouthLeft/Reshape%linear/linear_model/MouthLeft/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

>linear/linear_model/linear_model/linear_model/MouthRight/ShapeShapeParseExample/ParseExampleV2:5*
T0*
_output_shapes
:

Llinear/linear_model/linear_model/linear_model/MouthRight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Nlinear/linear_model/linear_model/linear_model/MouthRight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Nlinear/linear_model/linear_model/linear_model/MouthRight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ę
Flinear/linear_model/linear_model/linear_model/MouthRight/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/MouthRight/ShapeLlinear/linear_model/linear_model/linear_model/MouthRight/strided_slice/stackNlinear/linear_model/linear_model/linear_model/MouthRight/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/MouthRight/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Hlinear/linear_model/linear_model/linear_model/MouthRight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Flinear/linear_model/linear_model/linear_model/MouthRight/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/MouthRight/strided_sliceHlinear/linear_model/linear_model/linear_model/MouthRight/Reshape/shape/1*
N*
T0*
_output_shapes
:
ä
@linear/linear_model/linear_model/linear_model/MouthRight/ReshapeReshapeParseExample/ParseExampleV2:5Flinear/linear_model/linear_model/linear_model/MouthRight/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
5linear/linear_model/MouthRight/weights/ReadVariableOpReadVariableOp-linear/linear_model/MouthRight/weights/part_0*
_output_shapes

:*
dtype0

&linear/linear_model/MouthRight/weightsIdentity5linear/linear_model/MouthRight/weights/ReadVariableOp*
T0*
_output_shapes

:
ë
Elinear/linear_model/linear_model/linear_model/MouthRight/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/MouthRight/Reshape&linear/linear_model/MouthRight/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

<linear/linear_model/linear_model/linear_model/MouthTop/ShapeShapeParseExample/ParseExampleV2:6*
T0*
_output_shapes
:

Jlinear/linear_model/linear_model/linear_model/MouthTop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Llinear/linear_model/linear_model/linear_model/MouthTop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Llinear/linear_model/linear_model/linear_model/MouthTop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ŕ
Dlinear/linear_model/linear_model/linear_model/MouthTop/strided_sliceStridedSlice<linear/linear_model/linear_model/linear_model/MouthTop/ShapeJlinear/linear_model/linear_model/linear_model/MouthTop/strided_slice/stackLlinear/linear_model/linear_model/linear_model/MouthTop/strided_slice/stack_1Llinear/linear_model/linear_model/linear_model/MouthTop/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Flinear/linear_model/linear_model/linear_model/MouthTop/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Dlinear/linear_model/linear_model/linear_model/MouthTop/Reshape/shapePackDlinear/linear_model/linear_model/linear_model/MouthTop/strided_sliceFlinear/linear_model/linear_model/linear_model/MouthTop/Reshape/shape/1*
N*
T0*
_output_shapes
:
ŕ
>linear/linear_model/linear_model/linear_model/MouthTop/ReshapeReshapeParseExample/ParseExampleV2:6Dlinear/linear_model/linear_model/linear_model/MouthTop/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

3linear/linear_model/MouthTop/weights/ReadVariableOpReadVariableOp+linear/linear_model/MouthTop/weights/part_0*
_output_shapes

:*
dtype0

$linear/linear_model/MouthTop/weightsIdentity3linear/linear_model/MouthTop/weights/ReadVariableOp*
T0*
_output_shapes

:
ĺ
Clinear/linear_model/linear_model/linear_model/MouthTop/weighted_sumMatMul>linear/linear_model/linear_model/linear_model/MouthTop/Reshape$linear/linear_model/MouthTop/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

8linear/linear_model/linear_model/linear_model/Nose/ShapeShapeParseExample/ParseExampleV2:7*
T0*
_output_shapes
:

Flinear/linear_model/linear_model/linear_model/Nose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Hlinear/linear_model/linear_model/linear_model/Nose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Hlinear/linear_model/linear_model/linear_model/Nose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ź
@linear/linear_model/linear_model/linear_model/Nose/strided_sliceStridedSlice8linear/linear_model/linear_model/linear_model/Nose/ShapeFlinear/linear_model/linear_model/linear_model/Nose/strided_slice/stackHlinear/linear_model/linear_model/linear_model/Nose/strided_slice/stack_1Hlinear/linear_model/linear_model/linear_model/Nose/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Blinear/linear_model/linear_model/linear_model/Nose/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ü
@linear/linear_model/linear_model/linear_model/Nose/Reshape/shapePack@linear/linear_model/linear_model/linear_model/Nose/strided_sliceBlinear/linear_model/linear_model/linear_model/Nose/Reshape/shape/1*
N*
T0*
_output_shapes
:
Ř
:linear/linear_model/linear_model/linear_model/Nose/ReshapeReshapeParseExample/ParseExampleV2:7@linear/linear_model/linear_model/linear_model/Nose/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/linear/linear_model/Nose/weights/ReadVariableOpReadVariableOp'linear/linear_model/Nose/weights/part_0*
_output_shapes

:*
dtype0

 linear/linear_model/Nose/weightsIdentity/linear/linear_model/Nose/weights/ReadVariableOp*
T0*
_output_shapes

:
Ů
?linear/linear_model/linear_model/linear_model/Nose/weighted_sumMatMul:linear/linear_model/linear_model/linear_model/Nose/Reshape linear/linear_model/Nose/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=linear/linear_model/linear_model/linear_model/RightEye1/ShapeShapeParseExample/ParseExampleV2:8*
T0*
_output_shapes
:

Klinear/linear_model/linear_model/linear_model/RightEye1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Mlinear/linear_model/linear_model/linear_model/RightEye1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Mlinear/linear_model/linear_model/linear_model/RightEye1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ĺ
Elinear/linear_model/linear_model/linear_model/RightEye1/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/RightEye1/ShapeKlinear/linear_model/linear_model/linear_model/RightEye1/strided_slice/stackMlinear/linear_model/linear_model/linear_model/RightEye1/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/RightEye1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Glinear/linear_model/linear_model/linear_model/RightEye1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Elinear/linear_model/linear_model/linear_model/RightEye1/Reshape/shapePackElinear/linear_model/linear_model/linear_model/RightEye1/strided_sliceGlinear/linear_model/linear_model/linear_model/RightEye1/Reshape/shape/1*
N*
T0*
_output_shapes
:
â
?linear/linear_model/linear_model/linear_model/RightEye1/ReshapeReshapeParseExample/ParseExampleV2:8Elinear/linear_model/linear_model/linear_model/RightEye1/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
4linear/linear_model/RightEye1/weights/ReadVariableOpReadVariableOp,linear/linear_model/RightEye1/weights/part_0*
_output_shapes

:*
dtype0

%linear/linear_model/RightEye1/weightsIdentity4linear/linear_model/RightEye1/weights/ReadVariableOp*
T0*
_output_shapes

:
č
Dlinear/linear_model/linear_model/linear_model/RightEye1/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/RightEye1/Reshape%linear/linear_model/RightEye1/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=linear/linear_model/linear_model/linear_model/RightEye2/ShapeShapeParseExample/ParseExampleV2:9*
T0*
_output_shapes
:

Klinear/linear_model/linear_model/linear_model/RightEye2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Mlinear/linear_model/linear_model/linear_model/RightEye2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Mlinear/linear_model/linear_model/linear_model/RightEye2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ĺ
Elinear/linear_model/linear_model/linear_model/RightEye2/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/RightEye2/ShapeKlinear/linear_model/linear_model/linear_model/RightEye2/strided_slice/stackMlinear/linear_model/linear_model/linear_model/RightEye2/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/RightEye2/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Glinear/linear_model/linear_model/linear_model/RightEye2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :

Elinear/linear_model/linear_model/linear_model/RightEye2/Reshape/shapePackElinear/linear_model/linear_model/linear_model/RightEye2/strided_sliceGlinear/linear_model/linear_model/linear_model/RightEye2/Reshape/shape/1*
N*
T0*
_output_shapes
:
â
?linear/linear_model/linear_model/linear_model/RightEye2/ReshapeReshapeParseExample/ParseExampleV2:9Elinear/linear_model/linear_model/linear_model/RightEye2/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
4linear/linear_model/RightEye2/weights/ReadVariableOpReadVariableOp,linear/linear_model/RightEye2/weights/part_0*
_output_shapes

:*
dtype0

%linear/linear_model/RightEye2/weightsIdentity4linear/linear_model/RightEye2/weights/ReadVariableOp*
T0*
_output_shapes

:
č
Dlinear/linear_model/linear_model/linear_model/RightEye2/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/RightEye2/Reshape%linear/linear_model/RightEye2/weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Blinear/linear_model/linear_model/linear_model/weighted_sum_no_biasAddN?linear/linear_model/linear_model/linear_model/Chin/weighted_sumClinear/linear_model/linear_model/linear_model/LeftEye1/weighted_sumClinear/linear_model/linear_model/linear_model/LeftEye2/weighted_sumFlinear/linear_model/linear_model/linear_model/MouthBottom/weighted_sumDlinear/linear_model/linear_model/linear_model/MouthLeft/weighted_sumElinear/linear_model/linear_model/linear_model/MouthRight/weighted_sumClinear/linear_model/linear_model/linear_model/MouthTop/weighted_sum?linear/linear_model/linear_model/linear_model/Nose/weighted_sumDlinear/linear_model/linear_model/linear_model/RightEye1/weighted_sumDlinear/linear_model/linear_model/linear_model/RightEye2/weighted_sum*
N
*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/linear/linear_model/bias_weights/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0

 linear/linear_model/bias_weightsIdentity/linear/linear_model/bias_weights/ReadVariableOp*
T0*
_output_shapes
:
Ý
:linear/linear_model/linear_model/linear_model/weighted_sumBiasAddBlinear/linear_model/linear_model/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
linear/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0
d
linear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
f
linear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
f
linear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ů
linear/strided_sliceStridedSlicelinear/ReadVariableOplinear/strided_slice/stacklinear/strided_slice/stack_1linear/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
\
linear/bias/tagsConst*
_output_shapes
: *
dtype0*
valueB Blinear/bias
e
linear/biasScalarSummarylinear/bias/tagslinear/strided_slice*
T0*
_output_shapes
: 

3linear/zero_fraction/total_size/Size/ReadVariableOpReadVariableOp'linear/linear_model/Chin/weights/part_0*
_output_shapes

:*
dtype0
f
$linear/zero_fraction/total_size/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R
Ą
5linear/zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye1/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
Ą
5linear/zero_fraction/total_size/Size_2/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye2/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_2Const*
_output_shapes
: *
dtype0	*
value	B	 R
¤
5linear/zero_fraction/total_size/Size_3/ReadVariableOpReadVariableOp.linear/linear_model/MouthBottom/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_3Const*
_output_shapes
: *
dtype0	*
value	B	 R
˘
5linear/zero_fraction/total_size/Size_4/ReadVariableOpReadVariableOp,linear/linear_model/MouthLeft/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_4Const*
_output_shapes
: *
dtype0	*
value	B	 R
Ł
5linear/zero_fraction/total_size/Size_5/ReadVariableOpReadVariableOp-linear/linear_model/MouthRight/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_5Const*
_output_shapes
: *
dtype0	*
value	B	 R
Ą
5linear/zero_fraction/total_size/Size_6/ReadVariableOpReadVariableOp+linear/linear_model/MouthTop/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_6Const*
_output_shapes
: *
dtype0	*
value	B	 R

5linear/zero_fraction/total_size/Size_7/ReadVariableOpReadVariableOp'linear/linear_model/Nose/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_7Const*
_output_shapes
: *
dtype0	*
value	B	 R
˘
5linear/zero_fraction/total_size/Size_8/ReadVariableOpReadVariableOp,linear/linear_model/RightEye1/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_8Const*
_output_shapes
: *
dtype0	*
value	B	 R
˘
5linear/zero_fraction/total_size/Size_9/ReadVariableOpReadVariableOp,linear/linear_model/RightEye2/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_9Const*
_output_shapes
: *
dtype0	*
value	B	 R
ä
$linear/zero_fraction/total_size/AddNAddN$linear/zero_fraction/total_size/Size&linear/zero_fraction/total_size/Size_1&linear/zero_fraction/total_size/Size_2&linear/zero_fraction/total_size/Size_3&linear/zero_fraction/total_size/Size_4&linear/zero_fraction/total_size/Size_5&linear/zero_fraction/total_size/Size_6&linear/zero_fraction/total_size/Size_7&linear/zero_fraction/total_size/Size_8&linear/zero_fraction/total_size/Size_9*
N
*
T0	*
_output_shapes
: 
g
%linear/zero_fraction/total_zero/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 

%linear/zero_fraction/total_zero/EqualEqual$linear/zero_fraction/total_size/Size%linear/zero_fraction/total_zero/Const*
T0	*
_output_shapes
: 
Ů
*linear/zero_fraction/total_zero/zero_countIf%linear/zero_fraction/total_zero/Equal'linear/linear_model/Chin/weights/part_0$linear/zero_fraction/total_size/Size*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*H
else_branch9R7
5linear_zero_fraction_total_zero_zero_count_false_2550*
output_shapes
: *G
then_branch8R6
4linear_zero_fraction_total_zero_zero_count_true_2549

3linear/zero_fraction/total_zero/zero_count/IdentityIdentity*linear/zero_fraction/total_zero/zero_count*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_1Equal&linear/zero_fraction/total_size/Size_1'linear/zero_fraction/total_zero/Const_1*
T0	*
_output_shapes
: 
ç
,linear/zero_fraction/total_zero/zero_count_1If'linear/zero_fraction/total_zero/Equal_1+linear/linear_model/LeftEye1/weights/part_0&linear/zero_fraction/total_size/Size_1*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_1_false_2593*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_1_true_2592

5linear/zero_fraction/total_zero/zero_count_1/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_1*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_2Equal&linear/zero_fraction/total_size/Size_2'linear/zero_fraction/total_zero/Const_2*
T0	*
_output_shapes
: 
ç
,linear/zero_fraction/total_zero/zero_count_2If'linear/zero_fraction/total_zero/Equal_2+linear/linear_model/LeftEye2/weights/part_0&linear/zero_fraction/total_size/Size_2*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_2_false_2636*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_2_true_2635

5linear/zero_fraction/total_zero/zero_count_2/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_2*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_3Equal&linear/zero_fraction/total_size/Size_3'linear/zero_fraction/total_zero/Const_3*
T0	*
_output_shapes
: 
ę
,linear/zero_fraction/total_zero/zero_count_3If'linear/zero_fraction/total_zero/Equal_3.linear/linear_model/MouthBottom/weights/part_0&linear/zero_fraction/total_size/Size_3*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_3_false_2679*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_3_true_2678

5linear/zero_fraction/total_zero/zero_count_3/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_3*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_4Equal&linear/zero_fraction/total_size/Size_4'linear/zero_fraction/total_zero/Const_4*
T0	*
_output_shapes
: 
č
,linear/zero_fraction/total_zero/zero_count_4If'linear/zero_fraction/total_zero/Equal_4,linear/linear_model/MouthLeft/weights/part_0&linear/zero_fraction/total_size/Size_4*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_4_false_2722*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_4_true_2721

5linear/zero_fraction/total_zero/zero_count_4/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_4*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_5Equal&linear/zero_fraction/total_size/Size_5'linear/zero_fraction/total_zero/Const_5*
T0	*
_output_shapes
: 
é
,linear/zero_fraction/total_zero/zero_count_5If'linear/zero_fraction/total_zero/Equal_5-linear/linear_model/MouthRight/weights/part_0&linear/zero_fraction/total_size/Size_5*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_5_false_2765*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_5_true_2764

5linear/zero_fraction/total_zero/zero_count_5/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_5*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_6Equal&linear/zero_fraction/total_size/Size_6'linear/zero_fraction/total_zero/Const_6*
T0	*
_output_shapes
: 
ç
,linear/zero_fraction/total_zero/zero_count_6If'linear/zero_fraction/total_zero/Equal_6+linear/linear_model/MouthTop/weights/part_0&linear/zero_fraction/total_size/Size_6*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_6_false_2808*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_6_true_2807

5linear/zero_fraction/total_zero/zero_count_6/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_6*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_7Equal&linear/zero_fraction/total_size/Size_7'linear/zero_fraction/total_zero/Const_7*
T0	*
_output_shapes
: 
ă
,linear/zero_fraction/total_zero/zero_count_7If'linear/zero_fraction/total_zero/Equal_7'linear/linear_model/Nose/weights/part_0&linear/zero_fraction/total_size/Size_7*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_7_false_2851*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_7_true_2850

5linear/zero_fraction/total_zero/zero_count_7/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_7*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_8Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_8Equal&linear/zero_fraction/total_size/Size_8'linear/zero_fraction/total_zero/Const_8*
T0	*
_output_shapes
: 
č
,linear/zero_fraction/total_zero/zero_count_8If'linear/zero_fraction/total_zero/Equal_8,linear/linear_model/RightEye1/weights/part_0&linear/zero_fraction/total_size/Size_8*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_8_false_2894*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_8_true_2893

5linear/zero_fraction/total_zero/zero_count_8/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_8*
T0*
_output_shapes
: 
i
'linear/zero_fraction/total_zero/Const_9Const*
_output_shapes
: *
dtype0	*
value	B	 R 
˘
'linear/zero_fraction/total_zero/Equal_9Equal&linear/zero_fraction/total_size/Size_9'linear/zero_fraction/total_zero/Const_9*
T0	*
_output_shapes
: 
č
,linear/zero_fraction/total_zero/zero_count_9If'linear/zero_fraction/total_zero/Equal_9,linear/linear_model/RightEye2/weights/part_0&linear/zero_fraction/total_size/Size_9*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*J
else_branch;R9
7linear_zero_fraction_total_zero_zero_count_9_false_2937*
output_shapes
: *I
then_branch:R8
6linear_zero_fraction_total_zero_zero_count_9_true_2936

5linear/zero_fraction/total_zero/zero_count_9/IdentityIdentity,linear/zero_fraction/total_zero/zero_count_9*
T0*
_output_shapes
: 
ú
$linear/zero_fraction/total_zero/AddNAddN3linear/zero_fraction/total_zero/zero_count/Identity5linear/zero_fraction/total_zero/zero_count_1/Identity5linear/zero_fraction/total_zero/zero_count_2/Identity5linear/zero_fraction/total_zero/zero_count_3/Identity5linear/zero_fraction/total_zero/zero_count_4/Identity5linear/zero_fraction/total_zero/zero_count_5/Identity5linear/zero_fraction/total_zero/zero_count_6/Identity5linear/zero_fraction/total_zero/zero_count_7/Identity5linear/zero_fraction/total_zero/zero_count_8/Identity5linear/zero_fraction/total_zero/zero_count_9/Identity*
N
*
T0*
_output_shapes
: 

)linear/zero_fraction/compute/float32_sizeCast$linear/zero_fraction/total_size/AddN*

DstT0*

SrcT0	*
_output_shapes
: 
Ą
$linear/zero_fraction/compute/truedivRealDiv$linear/zero_fraction/total_zero/AddN)linear/zero_fraction/compute/float32_size*
T0*
_output_shapes
: 
|
)linear/zero_fraction/zero_fraction_or_nanIdentity$linear/zero_fraction/compute/truediv*
T0*
_output_shapes
: 

$linear/fraction_of_zero_weights/tagsConst*
_output_shapes
: *
dtype0*0
value'B% Blinear/fraction_of_zero_weights
˘
linear/fraction_of_zero_weightsScalarSummary$linear/fraction_of_zero_weights/tags)linear/zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 

$linear/head/predictions/logits/ShapeShape:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*
_output_shapes
:
z
8linear/head/predictions/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
j
blinear/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
[
Slinear/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

 linear/head/predictions/logisticSigmoid:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"linear/head/predictions/zeros_like	ZerosLike:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
-linear/head/predictions/two_class_logits/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
ţ
(linear/head/predictions/two_class_logitsConcatV2"linear/head/predictions/zeros_like:linear/linear_model/linear_model/linear_model/weighted_sum-linear/head/predictions/two_class_logits/axis*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%linear/head/predictions/probabilitiesSoftmax(linear/head/predictions/two_class_logits*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
+linear/head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
°
!linear/head/predictions/class_idsArgMax(linear/head/predictions/two_class_logits+linear/head/predictions/class_ids/dimension*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
&linear/head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
­
"linear/head/predictions/ExpandDims
ExpandDims!linear/head/predictions/class_ids&linear/head/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

linear/head/predictions/ShapeShape:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*
_output_shapes
:
u
+linear/head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
w
-linear/head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
w
-linear/head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ľ
%linear/head/predictions/strided_sliceStridedSlicelinear/head/predictions/Shape+linear/head/predictions/strided_slice/stack-linear/head/predictions/strided_slice/stack_1-linear/head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
e
#linear/head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
e
#linear/head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
e
#linear/head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ą
linear/head/predictions/rangeRange#linear/head/predictions/range/start#linear/head/predictions/range/limit#linear/head/predictions/range/delta*
_output_shapes
:
j
(linear/head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
¤
$linear/head/predictions/ExpandDims_1
ExpandDimslinear/head/predictions/range(linear/head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
j
(linear/head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
­
&linear/head/predictions/Tile/multiplesPack%linear/head/predictions/strided_slice(linear/head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:
¤
linear/head/predictions/TileTile$linear/head/predictions/ExpandDims_1&linear/head/predictions/Tile/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

linear/head/predictions/Shape_1Shape:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*
_output_shapes
:
w
-linear/head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
y
/linear/head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
y
/linear/head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ż
'linear/head/predictions/strided_slice_1StridedSlicelinear/head/predictions/Shape_1-linear/head/predictions/strided_slice_1/stack/linear/head/predictions/strided_slice_1/stack_1/linear/head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
g
%linear/head/predictions/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
g
%linear/head/predictions/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
g
%linear/head/predictions/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
š
linear/head/predictions/range_1Range%linear/head/predictions/range_1/start%linear/head/predictions/range_1/limit%linear/head/predictions/range_1/delta*
_output_shapes
:
r
 linear/head/predictions/AsStringAsStringlinear/head/predictions/range_1*
T0*
_output_shapes
:
j
(linear/head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
§
$linear/head/predictions/ExpandDims_2
ExpandDims linear/head/predictions/AsString(linear/head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
l
*linear/head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
ł
(linear/head/predictions/Tile_1/multiplesPack'linear/head/predictions/strided_slice_1*linear/head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:
¨
linear/head/predictions/Tile_1Tile$linear/head/predictions/ExpandDims_2(linear/head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

#linear/head/predictions/str_classesAsString"linear/head/predictions/ExpandDims*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
linear/head/ShapeShape%linear/head/predictions/probabilities*
T0*
_output_shapes
:
i
linear/head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!linear/head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!linear/head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
é
linear/head/strided_sliceStridedSlicelinear/head/Shapelinear/head/strided_slice/stack!linear/head/strided_slice/stack_1!linear/head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Y
linear/head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Y
linear/head/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
Y
linear/head/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :

linear/head/rangeRangelinear/head/range/startlinear/head/range/limitlinear/head/range/delta*
_output_shapes
:
X
linear/head/AsStringAsStringlinear/head/range*
T0*
_output_shapes
:
\
linear/head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

linear/head/ExpandDims
ExpandDimslinear/head/AsStringlinear/head/ExpandDims/dim*
T0*
_output_shapes

:
^
linear/head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :

linear/head/Tile/multiplesPacklinear/head/strided_slicelinear/head/Tile/multiples/1*
N*
T0*
_output_shapes
:
~
linear/head/TileTilelinear/head/ExpandDimslinear/head/Tile/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/Read/ReadVariableOpReadVariableOp'linear/linear_model/Chin/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
k
save/IdentityIdentitysave/Read/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
b
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes

:

save/Read_1/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye1/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
o
save/Identity_2Identitysave/Read_1/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:

save/Read_2/ReadVariableOpReadVariableOp+linear/linear_model/LeftEye2/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
o
save/Identity_4Identitysave/Read_2/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
d
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes

:

save/Read_3/ReadVariableOpReadVariableOp.linear/linear_model/MouthBottom/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
o
save/Identity_6Identitysave/Read_3/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:

save/Read_4/ReadVariableOpReadVariableOp,linear/linear_model/MouthLeft/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
o
save/Identity_8Identitysave/Read_4/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
d
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes

:

save/Read_5/ReadVariableOpReadVariableOp-linear/linear_model/MouthRight/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_10Identitysave/Read_5/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:

save/Read_6/ReadVariableOpReadVariableOp+linear/linear_model/MouthTop/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_12Identitysave/Read_6/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes

:

save/Read_7/ReadVariableOpReadVariableOp'linear/linear_model/Nose/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:

save/Read_8/ReadVariableOpReadVariableOp,linear/linear_model/RightEye1/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_16Identitysave/Read_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes

:

save/Read_9/ReadVariableOpReadVariableOp,linear/linear_model/RightEye2/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:

save/Read_10/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
T0*
_output_shapes
:

save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ŕ
valueśBłBglobal_stepB linear/linear_model/Chin/weightsB$linear/linear_model/LeftEye1/weightsB$linear/linear_model/LeftEye2/weightsB'linear/linear_model/MouthBottom/weightsB%linear/linear_model/MouthLeft/weightsB&linear/linear_model/MouthRight/weightsB$linear/linear_model/MouthTop/weightsB linear/linear_model/Nose/weightsB%linear/linear_model/RightEye1/weightsB%linear/linear_model/RightEye2/weightsB linear/linear_model/bias_weights

save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 0,1
đ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step/Read/ReadVariableOpsave/Identity_1save/Identity_3save/Identity_5save/Identity_7save/Identity_9save/Identity_11save/Identity_13save/Identity_15save/Identity_17save/Identity_19save/Identity_21"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/Identity_22Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ŕ
valueśBłBglobal_stepB linear/linear_model/Chin/weightsB$linear/linear_model/LeftEye1/weightsB$linear/linear_model/LeftEye2/weightsB'linear/linear_model/MouthBottom/weightsB%linear/linear_model/MouthLeft/weightsB&linear/linear_model/MouthRight/weightsB$linear/linear_model/MouthTop/weightsB linear/linear_model/Nose/weightsB%linear/linear_model/RightEye1/weightsB%linear/linear_model/RightEye2/weightsB linear/linear_model/bias_weights

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 0,1

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapesp
n::::::::::::*
dtypes
2	
O
save/Identity_23Identitysave/RestoreV2*
T0	*
_output_shapes
:
U
save/AssignVariableOpAssignVariableOpglobal_stepsave/Identity_23*
dtype0	
W
save/Identity_24Identitysave/RestoreV2:1*
T0*
_output_shapes

:
s
save/AssignVariableOp_1AssignVariableOp'linear/linear_model/Chin/weights/part_0save/Identity_24*
dtype0
W
save/Identity_25Identitysave/RestoreV2:2*
T0*
_output_shapes

:
w
save/AssignVariableOp_2AssignVariableOp+linear/linear_model/LeftEye1/weights/part_0save/Identity_25*
dtype0
W
save/Identity_26Identitysave/RestoreV2:3*
T0*
_output_shapes

:
w
save/AssignVariableOp_3AssignVariableOp+linear/linear_model/LeftEye2/weights/part_0save/Identity_26*
dtype0
W
save/Identity_27Identitysave/RestoreV2:4*
T0*
_output_shapes

:
z
save/AssignVariableOp_4AssignVariableOp.linear/linear_model/MouthBottom/weights/part_0save/Identity_27*
dtype0
W
save/Identity_28Identitysave/RestoreV2:5*
T0*
_output_shapes

:
x
save/AssignVariableOp_5AssignVariableOp,linear/linear_model/MouthLeft/weights/part_0save/Identity_28*
dtype0
W
save/Identity_29Identitysave/RestoreV2:6*
T0*
_output_shapes

:
y
save/AssignVariableOp_6AssignVariableOp-linear/linear_model/MouthRight/weights/part_0save/Identity_29*
dtype0
W
save/Identity_30Identitysave/RestoreV2:7*
T0*
_output_shapes

:
w
save/AssignVariableOp_7AssignVariableOp+linear/linear_model/MouthTop/weights/part_0save/Identity_30*
dtype0
W
save/Identity_31Identitysave/RestoreV2:8*
T0*
_output_shapes

:
s
save/AssignVariableOp_8AssignVariableOp'linear/linear_model/Nose/weights/part_0save/Identity_31*
dtype0
W
save/Identity_32Identitysave/RestoreV2:9*
T0*
_output_shapes

:
x
save/AssignVariableOp_9AssignVariableOp,linear/linear_model/RightEye1/weights/part_0save/Identity_32*
dtype0
X
save/Identity_33Identitysave/RestoreV2:10*
T0*
_output_shapes

:
y
save/AssignVariableOp_10AssignVariableOp,linear/linear_model/RightEye2/weights/part_0save/Identity_33*
dtype0
T
save/Identity_34Identitysave/RestoreV2:11*
T0*
_output_shapes
:
t
save/AssignVariableOp_11AssignVariableOp'linear/linear_model/bias_weights/part_0save/Identity_34*
dtype0
Ň
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard ň

e
4linear_zero_fraction_total_zero_zero_count_true_2549
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

g
6linear_zero_fraction_total_zero_zero_count_3_true_2678
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

g
6linear_zero_fraction_total_zero_zero_count_9_true_2936
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_29467
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_29047
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Š
Ĺ
7linear_zero_fraction_total_zero_zero_count_4_false_2722M
Izero_fraction_readvariableop_linear_linear_model_mouthleft_weights_part_0/
+cast_linear_zero_fraction_total_size_size_4	
mulĆ
zero_fraction/ReadVariableOpReadVariableOpIzero_fraction_readvariableop_linear_linear_model_mouthleft_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2732*
output_shapes
: */
then_branch R
zero_fraction_cond_true_27312
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_4*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_26887
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
§
Ä
7linear_zero_fraction_total_zero_zero_count_2_false_2636L
Hzero_fraction_readvariableop_linear_linear_model_lefteye2_weights_part_0/
+cast_linear_zero_fraction_total_size_size_2	
mulĹ
zero_fraction/ReadVariableOpReadVariableOpHzero_fraction_readvariableop_linear_linear_model_lefteye2_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2646*
output_shapes
: */
then_branch R
zero_fraction_cond_true_26452
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_2*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_27747
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_25607
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

g
6linear_zero_fraction_total_zero_zero_count_6_true_2807
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

g
6linear_zero_fraction_total_zero_zero_count_5_true_2764
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_27327
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_28187
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

g
6linear_zero_fraction_total_zero_zero_count_1_true_2592
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 

g
6linear_zero_fraction_total_zero_zero_count_2_true_2635
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_26457
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
ń
a
zero_fraction_cond_true_26027
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
­
Ç
7linear_zero_fraction_total_zero_zero_count_3_false_2679O
Kzero_fraction_readvariableop_linear_linear_model_mouthbottom_weights_part_0/
+cast_linear_zero_fraction_total_size_size_3	
mulČ
zero_fraction/ReadVariableOpReadVariableOpKzero_fraction_readvariableop_linear_linear_model_mouthbottom_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2689*
output_shapes
: */
then_branch R
zero_fraction_cond_true_26882
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_3*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_28607
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

g
6linear_zero_fraction_total_zero_zero_count_8_true_2893
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_29477
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_27757
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
Š
Ĺ
7linear_zero_fraction_total_zero_zero_count_8_false_2894M
Izero_fraction_readvariableop_linear_linear_model_righteye1_weights_part_0/
+cast_linear_zero_fraction_total_size_size_8	
mulĆ
zero_fraction/ReadVariableOpReadVariableOpIzero_fraction_readvariableop_linear_linear_model_righteye1_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2904*
output_shapes
: */
then_branch R
zero_fraction_cond_true_29032
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_8*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 

Ŕ
7linear_zero_fraction_total_zero_zero_count_7_false_2851H
Dzero_fraction_readvariableop_linear_linear_model_nose_weights_part_0/
+cast_linear_zero_fraction_total_size_size_7	
mulÁ
zero_fraction/ReadVariableOpReadVariableOpDzero_fraction_readvariableop_linear_linear_model_nose_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2861*
output_shapes
: */
then_branch R
zero_fraction_cond_true_28602
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_7*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_25597
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_28617
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
§
Ä
7linear_zero_fraction_total_zero_zero_count_6_false_2808L
Hzero_fraction_readvariableop_linear_linear_model_mouthtop_weights_part_0/
+cast_linear_zero_fraction_total_size_size_6	
mulĹ
zero_fraction/ReadVariableOpReadVariableOpHzero_fraction_readvariableop_linear_linear_model_mouthtop_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2818*
output_shapes
: */
then_branch R
zero_fraction_cond_true_28172
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_6*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_28177
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

g
6linear_zero_fraction_total_zero_zero_count_7_true_2850
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_26897
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:
§
Ä
7linear_zero_fraction_total_zero_zero_count_1_false_2593L
Hzero_fraction_readvariableop_linear_linear_model_lefteye1_weights_part_0/
+cast_linear_zero_fraction_total_size_size_1	
mulĹ
zero_fraction/ReadVariableOpReadVariableOpHzero_fraction_readvariableop_linear_linear_model_lefteye1_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2603*
output_shapes
: */
then_branch R
zero_fraction_cond_true_26022
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_1*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
Š
Ĺ
7linear_zero_fraction_total_zero_zero_count_9_false_2937M
Izero_fraction_readvariableop_linear_linear_model_righteye2_weights_part_0/
+cast_linear_zero_fraction_total_size_size_9	
mulĆ
zero_fraction/ReadVariableOpReadVariableOpIzero_fraction_readvariableop_linear_linear_model_righteye2_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2947*
output_shapes
: */
then_branch R
zero_fraction_cond_true_29462
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_9*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
Đ
y
zero_fraction_cond_false_26467
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:

g
6linear_zero_fraction_total_zero_zero_count_4_true_2721
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: :

_output_shapes
: 
Ť
Ć
7linear_zero_fraction_total_zero_zero_count_5_false_2765N
Jzero_fraction_readvariableop_linear_linear_model_mouthright_weights_part_0/
+cast_linear_zero_fraction_total_size_size_5	
mulÇ
zero_fraction/ReadVariableOpReadVariableOpJzero_fraction_readvariableop_linear_linear_model_mouthright_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2775*
output_shapes
: */
then_branch R
zero_fraction_cond_true_27742
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionq
CastCast+cast_linear_zero_fraction_total_size_size_5*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_29037
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:

ź
5linear_zero_fraction_total_zero_zero_count_false_2550H
Dzero_fraction_readvariableop_linear_linear_model_chin_weights_part_0-
)cast_linear_zero_fraction_total_size_size	
mulÁ
zero_fraction/ReadVariableOpReadVariableOpDzero_fraction_readvariableop_linear_linear_model_chin_weights_part_0*
_output_shapes

:*
dtype02
zero_fraction/ReadVariableOpj
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R˙˙˙˙2
zero_fraction/LessEqual/yĄ
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqualů
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
else_branch!R
zero_fraction_cond_false_2560*
output_shapes
: */
then_branch R
zero_fraction_cond_true_25592
zero_fraction/cond
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identityˇ
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub°
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast§
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1Ř
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractiono
CastCast)cast_linear_zero_fraction_total_size_size*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: :

_output_shapes
: 
ń
a
zero_fraction_cond_true_27317
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes

::$  

_output_shapes

:
Đ
y
zero_fraction_cond_false_26037
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros¸
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0*
_output_shapes

:2
count_nonzero/NotEqual
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
*
_output_shapes

:2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes

::$  

_output_shapes

:"ą?
save/Const:0save/Identity_22:0save/restore_all (5 @F8"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"A
	summaries4
2
linear/bias:0
!linear/fraction_of_zero_weights:0"ú
trainable_variablesâß

)linear/linear_model/Chin/weights/part_0:0.linear/linear_model/Chin/weights/part_0/Assign=linear/linear_model/Chin/weights/part_0/Read/ReadVariableOp:0".
 linear/linear_model/Chin/weights  "(2;linear/linear_model/Chin/weights/part_0/Initializer/zeros:08

-linear/linear_model/LeftEye1/weights/part_0:02linear/linear_model/LeftEye1/weights/part_0/AssignAlinear/linear_model/LeftEye1/weights/part_0/Read/ReadVariableOp:0"2
$linear/linear_model/LeftEye1/weights  "(2?linear/linear_model/LeftEye1/weights/part_0/Initializer/zeros:08

-linear/linear_model/LeftEye2/weights/part_0:02linear/linear_model/LeftEye2/weights/part_0/AssignAlinear/linear_model/LeftEye2/weights/part_0/Read/ReadVariableOp:0"2
$linear/linear_model/LeftEye2/weights  "(2?linear/linear_model/LeftEye2/weights/part_0/Initializer/zeros:08
Ž
0linear/linear_model/MouthBottom/weights/part_0:05linear/linear_model/MouthBottom/weights/part_0/AssignDlinear/linear_model/MouthBottom/weights/part_0/Read/ReadVariableOp:0"5
'linear/linear_model/MouthBottom/weights  "(2Blinear/linear_model/MouthBottom/weights/part_0/Initializer/zeros:08
¤
.linear/linear_model/MouthLeft/weights/part_0:03linear/linear_model/MouthLeft/weights/part_0/AssignBlinear/linear_model/MouthLeft/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/MouthLeft/weights  "(2@linear/linear_model/MouthLeft/weights/part_0/Initializer/zeros:08
Š
/linear/linear_model/MouthRight/weights/part_0:04linear/linear_model/MouthRight/weights/part_0/AssignClinear/linear_model/MouthRight/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/MouthRight/weights  "(2Alinear/linear_model/MouthRight/weights/part_0/Initializer/zeros:08

-linear/linear_model/MouthTop/weights/part_0:02linear/linear_model/MouthTop/weights/part_0/AssignAlinear/linear_model/MouthTop/weights/part_0/Read/ReadVariableOp:0"2
$linear/linear_model/MouthTop/weights  "(2?linear/linear_model/MouthTop/weights/part_0/Initializer/zeros:08

)linear/linear_model/Nose/weights/part_0:0.linear/linear_model/Nose/weights/part_0/Assign=linear/linear_model/Nose/weights/part_0/Read/ReadVariableOp:0".
 linear/linear_model/Nose/weights  "(2;linear/linear_model/Nose/weights/part_0/Initializer/zeros:08
¤
.linear/linear_model/RightEye1/weights/part_0:03linear/linear_model/RightEye1/weights/part_0/AssignBlinear/linear_model/RightEye1/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/RightEye1/weights  "(2@linear/linear_model/RightEye1/weights/part_0/Initializer/zeros:08
¤
.linear/linear_model/RightEye2/weights/part_0:03linear/linear_model/RightEye2/weights/part_0/AssignBlinear/linear_model/RightEye2/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/RightEye2/weights  "(2@linear/linear_model/RightEye2/weights/part_0/Initializer/zeros:08

)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"Ý
	variablesĎĚ
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H

)linear/linear_model/Chin/weights/part_0:0.linear/linear_model/Chin/weights/part_0/Assign=linear/linear_model/Chin/weights/part_0/Read/ReadVariableOp:0".
 linear/linear_model/Chin/weights  "(2;linear/linear_model/Chin/weights/part_0/Initializer/zeros:08

-linear/linear_model/LeftEye1/weights/part_0:02linear/linear_model/LeftEye1/weights/part_0/AssignAlinear/linear_model/LeftEye1/weights/part_0/Read/ReadVariableOp:0"2
$linear/linear_model/LeftEye1/weights  "(2?linear/linear_model/LeftEye1/weights/part_0/Initializer/zeros:08

-linear/linear_model/LeftEye2/weights/part_0:02linear/linear_model/LeftEye2/weights/part_0/AssignAlinear/linear_model/LeftEye2/weights/part_0/Read/ReadVariableOp:0"2
$linear/linear_model/LeftEye2/weights  "(2?linear/linear_model/LeftEye2/weights/part_0/Initializer/zeros:08
Ž
0linear/linear_model/MouthBottom/weights/part_0:05linear/linear_model/MouthBottom/weights/part_0/AssignDlinear/linear_model/MouthBottom/weights/part_0/Read/ReadVariableOp:0"5
'linear/linear_model/MouthBottom/weights  "(2Blinear/linear_model/MouthBottom/weights/part_0/Initializer/zeros:08
¤
.linear/linear_model/MouthLeft/weights/part_0:03linear/linear_model/MouthLeft/weights/part_0/AssignBlinear/linear_model/MouthLeft/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/MouthLeft/weights  "(2@linear/linear_model/MouthLeft/weights/part_0/Initializer/zeros:08
Š
/linear/linear_model/MouthRight/weights/part_0:04linear/linear_model/MouthRight/weights/part_0/AssignClinear/linear_model/MouthRight/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/MouthRight/weights  "(2Alinear/linear_model/MouthRight/weights/part_0/Initializer/zeros:08

-linear/linear_model/MouthTop/weights/part_0:02linear/linear_model/MouthTop/weights/part_0/AssignAlinear/linear_model/MouthTop/weights/part_0/Read/ReadVariableOp:0"2
$linear/linear_model/MouthTop/weights  "(2?linear/linear_model/MouthTop/weights/part_0/Initializer/zeros:08

)linear/linear_model/Nose/weights/part_0:0.linear/linear_model/Nose/weights/part_0/Assign=linear/linear_model/Nose/weights/part_0/Read/ReadVariableOp:0".
 linear/linear_model/Nose/weights  "(2;linear/linear_model/Nose/weights/part_0/Initializer/zeros:08
¤
.linear/linear_model/RightEye1/weights/part_0:03linear/linear_model/RightEye1/weights/part_0/AssignBlinear/linear_model/RightEye1/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/RightEye1/weights  "(2@linear/linear_model/RightEye1/weights/part_0/Initializer/zeros:08
¤
.linear/linear_model/RightEye2/weights/part_0:03linear/linear_model/RightEye2/weights/part_0/AssignBlinear/linear_model/RightEye2/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/RightEye2/weights  "(2@linear/linear_model/RightEye2/weights/part_0/Initializer/zeros:08

)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08*ĺ
classificationŇ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙4
classes)
linear/head/Tile:0˙˙˙˙˙˙˙˙˙H
scores>
'linear/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify*ů
predictí
5
examples)
input_example_tensor:0˙˙˙˙˙˙˙˙˙F
all_class_ids5
linear/head/predictions/Tile:0˙˙˙˙˙˙˙˙˙F
all_classes7
 linear/head/predictions/Tile_1:0˙˙˙˙˙˙˙˙˙H
	class_ids;
$linear/head/predictions/ExpandDims:0	˙˙˙˙˙˙˙˙˙G
classes<
%linear/head/predictions/str_classes:0˙˙˙˙˙˙˙˙˙E
logistic9
"linear/head/predictions/logistic:0˙˙˙˙˙˙˙˙˙]
logitsS
<linear/linear_model/linear_model/linear_model/weighted_sum:0˙˙˙˙˙˙˙˙˙O
probabilities>
'linear/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*Ś

regression
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙D
outputs9
"linear/head/predictions/logistic:0˙˙˙˙˙˙˙˙˙tensorflow/serving/regress*ć
serving_defaultŇ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙4
classes)
linear/head/Tile:0˙˙˙˙˙˙˙˙˙H
scores>
'linear/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify