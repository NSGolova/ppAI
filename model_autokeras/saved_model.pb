ЕЬ
Щъ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Л
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68А·
|
normalization/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namenormalization/mean
u
&normalization/mean/Read/ReadVariableOpReadVariableOpnormalization/mean*
_output_shapes
:*
dtype0
Д
normalization/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namenormalization/variance
}
*normalization/variance/Read/ReadVariableOpReadVariableOpnormalization/variance*
_output_shapes
:*
dtype0
z
normalization/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *$
shared_namenormalization/count
s
'normalization/count/Read/ReadVariableOpReadVariableOpnormalization/count*
_output_shapes
: *
dtype0	
ж
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!separable_conv2d/depthwise_kernel
Я
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*&
_output_shapes
:*
dtype0
ж
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!separable_conv2d/pointwise_kernel
Я
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*&
_output_shapes
:@*
dtype0
В
separable_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameseparable_conv2d/bias
{
)separable_conv2d/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias*
_output_shapes
:@*
dtype0
к
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d_1/depthwise_kernel
г
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*&
_output_shapes
:@*
dtype0
л
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*4
shared_name%#separable_conv2d_1/pointwise_kernel
д
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*'
_output_shapes
:@А*
dtype0
З
separable_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_1/bias
А
+separable_conv2d_1/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias*
_output_shapes	
:А*
dtype0
О
regression_head_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А└<*)
shared_nameregression_head_1/kernel
З
,regression_head_1/kernel/Read/ReadVariableOpReadVariableOpregression_head_1/kernel* 
_output_shapes
:
А└<*
dtype0
Д
regression_head_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameregression_head_1/bias
}
*regression_head_1/bias/Read/ReadVariableOpReadVariableOpregression_head_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
r
ConstConst*&
_output_shapes
:*
dtype0*-
value$B""3)5B╛В7CПG@
t
Const_1Const*&
_output_shapes
:*
dtype0*-
value$B""гa┬F	5─H╥вB

NoOpNoOp
б4
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*┌3
value╨3B═3 B╞3
╪
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

	optimizer
loss

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
'
#_self_saveable_object_factories* 
6
#_self_saveable_object_factories
	keras_api* 
у

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
#_self_saveable_object_factories
 	keras_api
!_adapt_function*
ы
"depthwise_kernel
#pointwise_kernel
$bias
#%_self_saveable_object_factories
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
ы
,depthwise_kernel
-pointwise_kernel
.bias
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
╩
#6_self_saveable_object_factories
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;_random_generator
<__call__
*=&call_and_return_all_conditional_losses* 
╩
#>_self_saveable_object_factories
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C_random_generator
D__call__
*E&call_and_return_all_conditional_losses* 
│
#F_self_saveable_object_factories
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
╦

Mkernel
Nbias
#O_self_saveable_object_factories
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*
* 
* 

Vserving_default* 
* 
R
0
1
2
"3
#4
$5
,6
-7
.8
M9
N10*
<
"0
#1
$2
,3
-4
.5
M6
N7*
* 
░
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
`Z
VARIABLE_VALUEnormalization/mean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEnormalization/variance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEnormalization/count5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
{u
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEseparable_conv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

"0
#1
$2*

"0
#1
$2*
* 
У
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
}w
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEseparable_conv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

,0
-1
.2*

,0
-1
.2*
* 
У
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
С
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
7	variables
8trainable_variables
9regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
С
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
?	variables
@trainable_variables
Aregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
С
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 
* 
* 
hb
VARIABLE_VALUEregression_head_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEregression_head_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

M0
N1*

M0
N1*
* 
У
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 

0
1
2*
C
0
1
2
3
4
5
6
7
	8*

z0
{1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	|total
	}count
~	variables
	keras_api*
M

Аtotal

Бcount
В
_fn_kwargs
Г	variables
Д	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

|0
}1*

~	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

А0
Б1*

Г	variables*
К
serving_default_input_1Placeholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
╩
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1ConstConst_1!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasregression_head_1/kernelregression_head_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_1044355
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
п
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&normalization/mean/Read/ReadVariableOp*normalization/variance/Read/ReadVariableOp'normalization/count/Read/ReadVariableOp5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOp,regression_head_1/kernel/Read/ReadVariableOp*regression_head_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_save_1044563
№
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenormalization/meannormalization/variancenormalization/count!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasregression_head_1/kernelregression_head_1/biastotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__traced_restore_1044618пТ
ТC
╟	
"__inference__wrapped_model_1043784
input_1
model_normalization_sub_y
model_normalization_sqrt_xY
?model_separable_conv2d_separable_conv2d_readvariableop_resource:[
Amodel_separable_conv2d_separable_conv2d_readvariableop_1_resource:@D
6model_separable_conv2d_biasadd_readvariableop_resource:@[
Amodel_separable_conv2d_1_separable_conv2d_readvariableop_resource:@^
Cmodel_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:@АG
8model_separable_conv2d_1_biasadd_readvariableop_resource:	АJ
6model_regression_head_1_matmul_readvariableop_resource:
А└<E
7model_regression_head_1_biasadd_readvariableop_resource:
identityИв.model/regression_head_1/BiasAdd/ReadVariableOpв-model/regression_head_1/MatMul/ReadVariableOpв-model/separable_conv2d/BiasAdd/ReadVariableOpв6model/separable_conv2d/separable_conv2d/ReadVariableOpв8model/separable_conv2d/separable_conv2d/ReadVariableOp_1в/model/separable_conv2d_1/BiasAdd/ReadVariableOpв8model/separable_conv2d_1/separable_conv2d/ReadVariableOpв:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1t
model/cast_to_float32/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:         ddУ
model/normalization/subSubmodel/cast_to_float32/Cast:y:0model_normalization_sub_y*
T0*/
_output_shapes
:         ddm
model/normalization/SqrtSqrtmodel_normalization_sqrt_x*
T0*&
_output_shapes
:b
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Э
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*&
_output_shapes
:Ю
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*/
_output_shapes
:         dd╛
6model/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp?model_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0┬
8model/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpAmodel_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype0Ж
-model/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            Ж
5model/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      О
1model/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/normalization/truediv:z:0>model/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^^*
paddingVALID*
strides
Т
'model/separable_conv2d/separable_conv2dConv2D:model/separable_conv2d/separable_conv2d/depthwise:output:0@model/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:         ^^@*
paddingVALID*
strides
а
-model/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╠
model/separable_conv2d/BiasAddBiasAdd0model/separable_conv2d/separable_conv2d:output:05model/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^^@Ж
model/separable_conv2d/ReluRelu'model/separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         ^^@┬
8model/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╟
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype0И
/model/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      И
7model/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ь
3model/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative)model/separable_conv2d/Relu:activations:0@model/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:         XX@*
paddingVALID*
strides
Щ
)model/separable_conv2d_1/separable_conv2dConv2D<model/separable_conv2d_1/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         XXА*
paddingVALID*
strides
е
/model/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╙
 model/separable_conv2d_1/BiasAddBiasAdd2model/separable_conv2d_1/separable_conv2d:output:07model/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         XXАЛ
model/separable_conv2d_1/ReluRelu)model/separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:         XXАК
model/dropout/IdentityIdentity+model/separable_conv2d_1/Relu:activations:0*
T0*0
_output_shapes
:         XXАА
model/dropout_1/IdentityIdentitymodel/dropout/Identity:output:0*
T0*0
_output_shapes
:         XXАd
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Х
model/flatten/ReshapeReshape!model/dropout_1/Identity:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:         А└<ж
-model/regression_head_1/MatMul/ReadVariableOpReadVariableOp6model_regression_head_1_matmul_readvariableop_resource* 
_output_shapes
:
А└<*
dtype0▒
model/regression_head_1/MatMulMatMulmodel/flatten/Reshape:output:05model/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         в
.model/regression_head_1/BiasAdd/ReadVariableOpReadVariableOp7model_regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╛
model/regression_head_1/BiasAddBiasAdd(model/regression_head_1/MatMul:product:06model/regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         w
IdentityIdentity(model/regression_head_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ї
NoOpNoOp/^model/regression_head_1/BiasAdd/ReadVariableOp.^model/regression_head_1/MatMul/ReadVariableOp.^model/separable_conv2d/BiasAdd/ReadVariableOp7^model/separable_conv2d/separable_conv2d/ReadVariableOp9^model/separable_conv2d/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_1/BiasAdd/ReadVariableOp9^model/separable_conv2d_1/separable_conv2d/ReadVariableOp;^model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 2`
.model/regression_head_1/BiasAdd/ReadVariableOp.model/regression_head_1/BiasAdd/ReadVariableOp2^
-model/regression_head_1/MatMul/ReadVariableOp-model/regression_head_1/MatMul/ReadVariableOp2^
-model/separable_conv2d/BiasAdd/ReadVariableOp-model/separable_conv2d/BiasAdd/ReadVariableOp2p
6model/separable_conv2d/separable_conv2d/ReadVariableOp6model/separable_conv2d/separable_conv2d/ReadVariableOp2t
8model/separable_conv2d/separable_conv2d/ReadVariableOp_18model/separable_conv2d/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_1/BiasAdd/ReadVariableOp/model/separable_conv2d_1/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_1/separable_conv2d/ReadVariableOp8model/separable_conv2d_1/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1:,(
&
_output_shapes
::,(
&
_output_shapes
:
√
b
D__inference_dropout_layer_call_and_return_conditional_losses_1043876

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         XXАd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         XXА"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
¤
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043883

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         XXАd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         XXА"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
Ц
b
)__inference_dropout_layer_call_fn_1044419

inputs
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1043992x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         XXА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
о
¤
'__inference_model_layer_call_fn_1044104
input_1
unknown
	unknown_0#
	unknown_1:#
	unknown_2:@
	unknown_3:@#
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:
А└<
	unknown_8:
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1044056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1:,(
&
_output_shapes
::,(
&
_output_shapes
:
╟M
·
B__inference_model_layer_call_and_return_conditional_losses_1044328

inputs
normalization_sub_y
normalization_sqrt_xS
9separable_conv2d_separable_conv2d_readvariableop_resource:U
;separable_conv2d_separable_conv2d_readvariableop_1_resource:@>
0separable_conv2d_biasadd_readvariableop_resource:@U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:@X
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:@АA
2separable_conv2d_1_biasadd_readvariableop_resource:	АD
0regression_head_1_matmul_readvariableop_resource:
А└<?
1regression_head_1_biasadd_readvariableop_resource:
identityИв(regression_head_1/BiasAdd/ReadVariableOpв'regression_head_1/MatMul/ReadVariableOpв'separable_conv2d/BiasAdd/ReadVariableOpв0separable_conv2d/separable_conv2d/ReadVariableOpв2separable_conv2d/separable_conv2d/ReadVariableOp_1в)separable_conv2d_1/BiasAdd/ReadVariableOpв2separable_conv2d_1/separable_conv2d/ReadVariableOpв4separable_conv2d_1/separable_conv2d/ReadVariableOp_1m
cast_to_float32/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         ddБ
normalization/subSubcast_to_float32/Cast:y:0normalization_sub_y*
T0*/
_output_shapes
:         dda
normalization/SqrtSqrtnormalization_sqrt_x*
T0*&
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:М
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:         dd▓
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╢
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype0А
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      №
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativenormalization/truediv:z:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^^*
paddingVALID*
strides
А
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:         ^^@*
paddingVALID*
strides
Ф
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0║
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^^@z
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         ^^@╢
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╗
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype0В
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      В
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      К
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:         XX@*
paddingVALID*
strides
З
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         XXА*
paddingVALID*
strides
Щ
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0┴
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         XXА
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:         XXАZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ь
dropout/dropout/MulMul%separable_conv2d_1/Relu:activations:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         XXАj
dropout/dropout/ShapeShape%separable_conv2d_1/Relu:activations:0*
T0*
_output_shapes
:е
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         XXА*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╟
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         XXАИ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         XXАК
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         XXА\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Ф
dropout_1/dropout/MulMuldropout/dropout/Mul_1:z:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:         XXА`
dropout_1/dropout/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:й
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:         XXА*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>═
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         XXАМ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         XXАР
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:         XXА^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Г
flatten/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:         А└<Ъ
'regression_head_1/MatMul/ReadVariableOpReadVariableOp0regression_head_1_matmul_readvariableop_resource* 
_output_shapes
:
А└<*
dtype0Я
regression_head_1/MatMulMatMulflatten/Reshape:output:0/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
(regression_head_1/BiasAdd/ReadVariableOpReadVariableOp1regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
regression_head_1/BiasAddBiasAdd"regression_head_1/MatMul:product:00regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         q
IdentityIdentity"regression_head_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ┼
NoOpNoOp)^regression_head_1/BiasAdd/ReadVariableOp(^regression_head_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 2T
(regression_head_1/BiasAdd/ReadVariableOp(regression_head_1/BiasAdd/ReadVariableOp2R
'regression_head_1/MatMul/ReadVariableOp'regression_head_1/MatMul/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
╝

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043969

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         XXАC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         XXА*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         XXАx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         XXАr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         XXАb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         XXА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
м
Д
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1044382

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ┘
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╢
И
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1044409

inputsB
(separable_conv2d_readvariableop_resource:@E
*separable_conv2d_readvariableop_1_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Х
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ┘
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
р
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           Ае
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╩
`
D__inference_flatten_layer_call_and_return_conditional_losses_1043891

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         А└<Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         А└<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
║

c
D__inference_dropout_layer_call_and_return_conditional_losses_1043992

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         XXАC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         XXА*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         XXАx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         XXАr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         XXАb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         XXА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
▀
в
3__inference_regression_head_1_layer_call_fn_1044483

inputs
unknown:
А└<
	unknown_0:
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1043903o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         А└<: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         А└<
 
_user_specified_nameinputs
╚
G
+__inference_dropout_1_layer_call_fn_1044441

inputs
identity╜
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043883i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         XXА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
¤
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1044451

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         XXАd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         XXА"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
М
√
%__inference_signature_wrapper_1044355
input_1
unknown
	unknown_0#
	unknown_1:#
	unknown_2:@
	unknown_3:@#
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:
А└<
	unknown_8:
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_1043784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1:,(
&
_output_shapes
::,(
&
_output_shapes
:
╜(
▐
B__inference_model_layer_call_and_return_conditional_losses_1044056

inputs
normalization_sub_y
normalization_sqrt_x2
separable_conv2d_1044033:2
separable_conv2d_1044035:@&
separable_conv2d_1044037:@4
separable_conv2d_1_1044040:@5
separable_conv2d_1_1044042:@А)
separable_conv2d_1_1044044:	А-
regression_head_1_1044050:
А└<'
regression_head_1_1044052:
identityИвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallm
cast_to_float32/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         ddБ
normalization/subSubcast_to_float32/Cast:y:0normalization_sub_y*
T0*/
_output_shapes
:         dda
normalization/SqrtSqrtnormalization_sqrt_x*
T0*&
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:М
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:         dd═
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0separable_conv2d_1044033separable_conv2d_1044035separable_conv2d_1044037*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ^^@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1043804Ё
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_1044040separable_conv2d_1_1044042separable_conv2d_1_1044044*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1043833А
dropout/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1043992Ы
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043969р
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А└<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1043891┤
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regression_head_1_1044050regression_head_1_1044052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1043903Б
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Р
NoOpNoOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
П(
╙
__inference_adapt_step_1041260
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вIteratorGetNextвReadVariableOpвReadVariableOp_1вReadVariableOp_2вadd/ReadVariableOp┴
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*/
_output_shapes
:         dd*.
output_shapes
:         dd*
output_types
2s
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Э
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:е
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*/
_output_shapes
:         ddw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ж
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 a
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	e
GatherV2/indicesConst*
_output_shapes
:*
dtype0*!
valueB"          O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:П
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0В
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0Д
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
┘	
Б
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1044493

inputs2
matmul_readvariableop_resource:
А└<-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А└<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         А└<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         А└<
 
_user_specified_nameinputs
л
№
'__inference_model_layer_call_fn_1044197

inputs
unknown
	unknown_0#
	unknown_1:#
	unknown_2:@
	unknown_3:@#
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:
А└<
	unknown_8:
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1043910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ъ
d
+__inference_dropout_1_layer_call_fn_1044446

inputs
identityИвStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043969x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         XXА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
╩
`
D__inference_flatten_layer_call_and_return_conditional_losses_1044474

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         А└<Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         А└<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
м
Д
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1043804

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ┘
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╢
И
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1043833

inputsB
(separable_conv2d_readvariableop_resource:@E
*separable_conv2d_readvariableop_1_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Х
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ┘
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
р
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ъ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Аk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           А|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           Ае
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╓%
Щ
B__inference_model_layer_call_and_return_conditional_losses_1044138
input_1
normalization_sub_y
normalization_sqrt_x2
separable_conv2d_1044115:2
separable_conv2d_1044117:@&
separable_conv2d_1044119:@4
separable_conv2d_1_1044122:@5
separable_conv2d_1_1044124:@А)
separable_conv2d_1_1044126:	А-
regression_head_1_1044132:
А└<'
regression_head_1_1044134:
identityИв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCalln
cast_to_float32/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:         ddБ
normalization/subSubcast_to_float32/Cast:y:0normalization_sub_y*
T0*/
_output_shapes
:         dda
normalization/SqrtSqrtnormalization_sqrt_x*
T0*&
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:М
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:         dd═
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0separable_conv2d_1044115separable_conv2d_1044117separable_conv2d_1044119*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ^^@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1043804Ё
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_1044122separable_conv2d_1_1044124separable_conv2d_1_1044126*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1043833Ё
dropout/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1043876с
dropout_1/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043883╪
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А└<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1043891┤
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regression_head_1_1044132regression_head_1_1044134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1043903Б
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╩
NoOpNoOp*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1:,(
&
_output_shapes
::,(
&
_output_shapes
:
┘	
Б
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1043903

inputs2
matmul_readvariableop_resource:
А└<-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А└<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         А└<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         А└<
 
_user_specified_nameinputs
√
b
D__inference_dropout_layer_call_and_return_conditional_losses_1044424

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         XXАd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         XXА"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
Д	
╨
4__inference_separable_conv2d_1_layer_call_fn_1044393

inputs!
unknown:@$
	unknown_0:@А
	unknown_1:	А
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1043833К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
└(
▀
B__inference_model_layer_call_and_return_conditional_losses_1044172
input_1
normalization_sub_y
normalization_sqrt_x2
separable_conv2d_1044149:2
separable_conv2d_1044151:@&
separable_conv2d_1044153:@4
separable_conv2d_1_1044156:@5
separable_conv2d_1_1044158:@А)
separable_conv2d_1_1044160:	А-
regression_head_1_1044166:
А└<'
regression_head_1_1044168:
identityИвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCalln
cast_to_float32/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:         ddБ
normalization/subSubcast_to_float32/Cast:y:0normalization_sub_y*
T0*/
_output_shapes
:         dda
normalization/SqrtSqrtnormalization_sqrt_x*
T0*&
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:М
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:         dd═
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0separable_conv2d_1044149separable_conv2d_1044151separable_conv2d_1044153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ^^@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1043804Ё
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_1044156separable_conv2d_1_1044158separable_conv2d_1_1044160*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1043833А
dropout/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1043992Ы
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043969р
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А└<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1043891┤
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regression_head_1_1044166regression_head_1_1044168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1043903Б
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Р
NoOpNoOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1:,(
&
_output_shapes
::,(
&
_output_shapes
:
╒)
У
 __inference__traced_save_1044563
file_prefix1
-savev2_normalization_mean_read_readvariableop5
1savev2_normalization_variance_read_readvariableop2
.savev2_normalization_count_read_readvariableop	@
<savev2_separable_conv2d_depthwise_kernel_read_readvariableop@
<savev2_separable_conv2d_pointwise_kernel_read_readvariableop4
0savev2_separable_conv2d_bias_read_readvariableopB
>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_1_bias_read_readvariableop7
3savev2_regression_head_1_kernel_read_readvariableop5
1savev2_regression_head_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const_2

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ы
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ф
valueКBЗB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHН
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B й
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_normalization_mean_read_readvariableop1savev2_normalization_variance_read_readvariableop.savev2_normalization_count_read_readvariableop<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop3savev2_regression_head_1_kernel_read_readvariableop1savev2_regression_head_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ч
_input_shapesЕ
В: ::: ::@:@:@:@А:А:
А└<:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :,(
&
_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@:-)
'
_output_shapes
:@А:!	

_output_shapes	
:А:&
"
 
_output_shapes
:
А└<: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
№
╠
2__inference_separable_conv2d_layer_call_fn_1044366

inputs!
unknown:#
	unknown_0:@
	unknown_1:@
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1043804Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╢
E
)__inference_flatten_layer_call_fn_1044468

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А└<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1043891b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         А└<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
╙%
Ш
B__inference_model_layer_call_and_return_conditional_losses_1043910

inputs
normalization_sub_y
normalization_sqrt_x2
separable_conv2d_1043857:2
separable_conv2d_1043859:@&
separable_conv2d_1043861:@4
separable_conv2d_1_1043864:@5
separable_conv2d_1_1043866:@А)
separable_conv2d_1_1043868:	А-
regression_head_1_1043904:
А└<'
regression_head_1_1043906:
identityИв)regression_head_1/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallm
cast_to_float32/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         ddБ
normalization/subSubcast_to_float32/Cast:y:0normalization_sub_y*
T0*/
_output_shapes
:         dda
normalization/SqrtSqrtnormalization_sqrt_x*
T0*&
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:М
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:         dd═
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0separable_conv2d_1043857separable_conv2d_1043859separable_conv2d_1043861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ^^@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1043804Ё
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_1043864separable_conv2d_1_1043866separable_conv2d_1_1043868*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1043833Ё
dropout/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1043876с
dropout_1/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1043883╪
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А└<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1043891┤
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regression_head_1_1043904regression_head_1_1043906*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1043903Б
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╩
NoOpNoOp*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
л
№
'__inference_model_layer_call_fn_1044222

inputs
unknown
	unknown_0#
	unknown_1:#
	unknown_2:@
	unknown_3:@#
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:
А└<
	unknown_8:
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1044056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Л>
·
B__inference_model_layer_call_and_return_conditional_losses_1044268

inputs
normalization_sub_y
normalization_sqrt_xS
9separable_conv2d_separable_conv2d_readvariableop_resource:U
;separable_conv2d_separable_conv2d_readvariableop_1_resource:@>
0separable_conv2d_biasadd_readvariableop_resource:@U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:@X
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:@АA
2separable_conv2d_1_biasadd_readvariableop_resource:	АD
0regression_head_1_matmul_readvariableop_resource:
А└<?
1regression_head_1_biasadd_readvariableop_resource:
identityИв(regression_head_1/BiasAdd/ReadVariableOpв'regression_head_1/MatMul/ReadVariableOpв'separable_conv2d/BiasAdd/ReadVariableOpв0separable_conv2d/separable_conv2d/ReadVariableOpв2separable_conv2d/separable_conv2d/ReadVariableOp_1в)separable_conv2d_1/BiasAdd/ReadVariableOpв2separable_conv2d_1/separable_conv2d/ReadVariableOpв4separable_conv2d_1/separable_conv2d/ReadVariableOp_1m
cast_to_float32/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         ddБ
normalization/subSubcast_to_float32/Cast:y:0normalization_sub_y*
T0*/
_output_shapes
:         dda
normalization/SqrtSqrtnormalization_sqrt_x*
T0*&
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:М
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:         dd▓
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╢
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype0А
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      №
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativenormalization/truediv:z:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^^*
paddingVALID*
strides
А
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:         ^^@*
paddingVALID*
strides
Ф
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0║
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ^^@z
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         ^^@╢
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╗
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype0В
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      В
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      К
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:         XX@*
paddingVALID*
strides
З
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:         XXА*
paddingVALID*
strides
Щ
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0┴
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         XXА
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:         XXА~
dropout/IdentityIdentity%separable_conv2d_1/Relu:activations:0*
T0*0
_output_shapes
:         XXАt
dropout_1/IdentityIdentitydropout/Identity:output:0*
T0*0
_output_shapes
:         XXА^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Г
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         А└<Ъ
'regression_head_1/MatMul/ReadVariableOpReadVariableOp0regression_head_1_matmul_readvariableop_resource* 
_output_shapes
:
А└<*
dtype0Я
regression_head_1/MatMulMatMulflatten/Reshape:output:0/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
(regression_head_1/BiasAdd/ReadVariableOpReadVariableOp1regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
regression_head_1/BiasAddBiasAdd"regression_head_1/MatMul:product:00regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         q
IdentityIdentity"regression_head_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ┼
NoOpNoOp)^regression_head_1/BiasAdd/ReadVariableOp(^regression_head_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 2T
(regression_head_1/BiasAdd/ReadVariableOp(regression_head_1/BiasAdd/ReadVariableOp2R
'regression_head_1/MatMul/ReadVariableOp'regression_head_1/MatMul/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
╫?
ё	
#__inference__traced_restore_1044618
file_prefix1
#assignvariableop_normalization_mean:7
)assignvariableop_1_normalization_variance:0
&assignvariableop_2_normalization_count:	 N
4assignvariableop_3_separable_conv2d_depthwise_kernel:N
4assignvariableop_4_separable_conv2d_pointwise_kernel:@6
(assignvariableop_5_separable_conv2d_bias:@P
6assignvariableop_6_separable_conv2d_1_depthwise_kernel:@Q
6assignvariableop_7_separable_conv2d_1_pointwise_kernel:@А9
*assignvariableop_8_separable_conv2d_1_bias:	А?
+assignvariableop_9_regression_head_1_kernel:
А└<8
*assignvariableop_10_regression_head_1_bias:#
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: 
identity_16ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ю
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ф
valueКBЗB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHР
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B ю
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOpAssignVariableOp#assignvariableop_normalization_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_1AssignVariableOp)assignvariableop_1_normalization_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:Х
AssignVariableOp_2AssignVariableOp&assignvariableop_2_normalization_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_3AssignVariableOp4assignvariableop_3_separable_conv2d_depthwise_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_4AssignVariableOp4assignvariableop_4_separable_conv2d_pointwise_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_5AssignVariableOp(assignvariableop_5_separable_conv2d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_6AssignVariableOp6assignvariableop_6_separable_conv2d_1_depthwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_7AssignVariableOp6assignvariableop_7_separable_conv2d_1_pointwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_8AssignVariableOp*assignvariableop_8_separable_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_9AssignVariableOp+assignvariableop_9_regression_head_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_10AssignVariableOp*assignvariableop_10_regression_head_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Щ
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: Ж
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_16Identity_16:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
║

c
D__inference_dropout_layer_call_and_return_conditional_losses_1044436

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         XXАC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         XXА*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         XXАx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         XXАr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         XXАb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         XXА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
о
¤
'__inference_model_layer_call_fn_1043933
input_1
unknown
	unknown_0#
	unknown_1:#
	unknown_2:@
	unknown_3:@#
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:
А└<
	unknown_8:
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1043910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         dd::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1:,(
&
_output_shapes
::,(
&
_output_shapes
:
╝

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1044463

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         XXАC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         XXА*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         XXАx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         XXАr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         XXАb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         XXА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs
─
E
)__inference_dropout_layer_call_fn_1044414

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         XXА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1043876i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         XXА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         XXА:X T
0
_output_shapes
:         XXА
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╝
serving_defaultи
C
input_18
serving_default_input_1:0         ddE
regression_head_10
StatefulPartitionedCall:0         tensorflow/serving/predict:єЙ
я
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

	optimizer
loss

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
M
#_self_saveable_object_factories
	keras_api"
_tf_keras_layer
°

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
#_self_saveable_object_factories
 	keras_api
!_adapt_function"
_tf_keras_layer
А
"depthwise_kernel
#pointwise_kernel
$bias
#%_self_saveable_object_factories
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
А
,depthwise_kernel
-pointwise_kernel
.bias
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
с
#6_self_saveable_object_factories
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;_random_generator
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
с
#>_self_saveable_object_factories
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C_random_generator
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
╩
#F_self_saveable_object_factories
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
р

Mkernel
Nbias
#O_self_saveable_object_factories
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
 "
trackable_dict_wrapper
,
Vserving_default"
signature_map
 "
trackable_dict_wrapper
n
0
1
2
"3
#4
$5
,6
-7
.8
M9
N10"
trackable_list_wrapper
X
"0
#1
$2
,3
-4
.5
M6
N7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ъ2ч
'__inference_model_layer_call_fn_1043933
'__inference_model_layer_call_fn_1044197
'__inference_model_layer_call_fn_1044222
'__inference_model_layer_call_fn_1044104└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
B__inference_model_layer_call_and_return_conditional_losses_1044268
B__inference_model_layer_call_and_return_conditional_losses_1044328
B__inference_model_layer_call_and_return_conditional_losses_1044138
B__inference_model_layer_call_and_return_conditional_losses_1044172└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
═B╩
"__inference__wrapped_model_1043784input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2normalization/mean
": 2normalization/variance
:	 2normalization/count
 "
trackable_dict_wrapper
"
_generic_user_object
└2╜
__inference_adapt_step_1041260Ъ
У▓П
FullArgSpec
argsЪ

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
;:92!separable_conv2d/depthwise_kernel
;:9@2!separable_conv2d/pointwise_kernel
#:!@2separable_conv2d/bias
 "
trackable_dict_wrapper
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
▄2┘
2__inference_separable_conv2d_layer_call_fn_1044366в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ў2Ї
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1044382в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
=:;@2#separable_conv2d_1/depthwise_kernel
>:<@А2#separable_conv2d_1/pointwise_kernel
&:$А2separable_conv2d_1/bias
 "
trackable_dict_wrapper
5
,0
-1
.2"
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
▐2█
4__inference_separable_conv2d_1_layer_call_fn_1044393в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙2Ў
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1044409в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
7	variables
8trainable_variables
9regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Р2Н
)__inference_dropout_layer_call_fn_1044414
)__inference_dropout_layer_call_fn_1044419┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_layer_call_and_return_conditional_losses_1044424
D__inference_dropout_layer_call_and_return_conditional_losses_1044436┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
?	variables
@trainable_variables
Aregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_1_layer_call_fn_1044441
+__inference_dropout_1_layer_call_fn_1044446┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_1_layer_call_and_return_conditional_losses_1044451
F__inference_dropout_1_layer_call_and_return_conditional_losses_1044463┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_flatten_layer_call_fn_1044468в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_flatten_layer_call_and_return_conditional_losses_1044474в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,:*
А└<2regression_head_1/kernel
$:"2regression_head_1/bias
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
▌2┌
3__inference_regression_head_1_layer_call_fn_1044483в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°2ї
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1044493в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠B╔
%__inference_signature_wrapper_1044355input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
5
0
1
2"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	|total
	}count
~	variables
	keras_api"
_tf_keras_metric
c

Аtotal

Бcount
В
_fn_kwargs
Г	variables
Д	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
|0
}1"
trackable_list_wrapper
-
~	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
А0
Б1"
trackable_list_wrapper
.
Г	variables"
_generic_user_object
	J
Const
J	
Const_1╢
"__inference__wrapped_model_1043784ПЕЖ"#$,-.MN8в5
.в+
)К&
input_1         dd
к "EкB
@
regression_head_1+К(
regression_head_1         x
__inference_adapt_step_1041260VKвH
Aв>
<Т9%в"
 К         ddIteratorSpec 
к "
 ╕
F__inference_dropout_1_layer_call_and_return_conditional_losses_1044451n<в9
2в/
)К&
inputs         XXА
p 
к ".в+
$К!
0         XXА
Ъ ╕
F__inference_dropout_1_layer_call_and_return_conditional_losses_1044463n<в9
2в/
)К&
inputs         XXА
p
к ".в+
$К!
0         XXА
Ъ Р
+__inference_dropout_1_layer_call_fn_1044441a<в9
2в/
)К&
inputs         XXА
p 
к "!К         XXАР
+__inference_dropout_1_layer_call_fn_1044446a<в9
2в/
)К&
inputs         XXА
p
к "!К         XXА╢
D__inference_dropout_layer_call_and_return_conditional_losses_1044424n<в9
2в/
)К&
inputs         XXА
p 
к ".в+
$К!
0         XXА
Ъ ╢
D__inference_dropout_layer_call_and_return_conditional_losses_1044436n<в9
2в/
)К&
inputs         XXА
p
к ".в+
$К!
0         XXА
Ъ О
)__inference_dropout_layer_call_fn_1044414a<в9
2в/
)К&
inputs         XXА
p 
к "!К         XXАО
)__inference_dropout_layer_call_fn_1044419a<в9
2в/
)К&
inputs         XXА
p
к "!К         XXАл
D__inference_flatten_layer_call_and_return_conditional_losses_1044474c8в5
.в+
)К&
inputs         XXА
к "'в$
К
0         А└<
Ъ Г
)__inference_flatten_layer_call_fn_1044468V8в5
.в+
)К&
inputs         XXА
к "К         А└<╜
B__inference_model_layer_call_and_return_conditional_losses_1044138wЕЖ"#$,-.MN@в=
6в3
)К&
input_1         dd
p 

 
к "%в"
К
0         
Ъ ╜
B__inference_model_layer_call_and_return_conditional_losses_1044172wЕЖ"#$,-.MN@в=
6в3
)К&
input_1         dd
p

 
к "%в"
К
0         
Ъ ╝
B__inference_model_layer_call_and_return_conditional_losses_1044268vЕЖ"#$,-.MN?в<
5в2
(К%
inputs         dd
p 

 
к "%в"
К
0         
Ъ ╝
B__inference_model_layer_call_and_return_conditional_losses_1044328vЕЖ"#$,-.MN?в<
5в2
(К%
inputs         dd
p

 
к "%в"
К
0         
Ъ Х
'__inference_model_layer_call_fn_1043933jЕЖ"#$,-.MN@в=
6в3
)К&
input_1         dd
p 

 
к "К         Х
'__inference_model_layer_call_fn_1044104jЕЖ"#$,-.MN@в=
6в3
)К&
input_1         dd
p

 
к "К         Ф
'__inference_model_layer_call_fn_1044197iЕЖ"#$,-.MN?в<
5в2
(К%
inputs         dd
p 

 
к "К         Ф
'__inference_model_layer_call_fn_1044222iЕЖ"#$,-.MN?в<
5в2
(К%
inputs         dd
p

 
к "К         ░
N__inference_regression_head_1_layer_call_and_return_conditional_losses_1044493^MN1в.
'в$
"К
inputs         А└<
к "%в"
К
0         
Ъ И
3__inference_regression_head_1_layer_call_fn_1044483QMN1в.
'в$
"К
inputs         А└<
к "К         ц
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_1044409Т,-.IвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ ╛
4__inference_separable_conv2d_1_layer_call_fn_1044393Е,-.IвF
?в<
:К7
inputs+                           @
к "3К0,                           Ау
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_1044382С"#$IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           @
Ъ ╗
2__inference_separable_conv2d_layer_call_fn_1044366Д"#$IвF
?в<
:К7
inputs+                           
к "2К/+                           @─
%__inference_signature_wrapper_1044355ЪЕЖ"#$,-.MNCв@
в 
9к6
4
input_1)К&
input_1         dd"EкB
@
regression_head_1+К(
regression_head_1         