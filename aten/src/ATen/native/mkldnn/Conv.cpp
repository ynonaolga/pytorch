#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/Functions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/_add_relu_native.h>
#include <ATen/ops/conv2d.h>
#include <ATen/ops/conv3d.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mkldnn_convolution_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

REGISTER_NO_CPU_DISPATCH(mkldnn_convolution_backward_stub);

}}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

// follow check rules from native/Convolution.cpp without transpose supported
static void check_shape_forward(const Tensor& input,
                                const Tensor& weight,
                                const Tensor& bias,
                                const IntArrayRef& padding,
                                const IntArrayRef& stride,
                                const IntArrayRef& dilation,
                                const int64_t groups) {
#define MKLDNN_CONV_ARG_CHECK(IT, OP) std::any_of(IT.begin(), IT.end(), [](auto x) { return x OP 0; })
  auto is_padding_neg = MKLDNN_CONV_ARG_CHECK(padding, <);
  auto is_stride_nonpos = MKLDNN_CONV_ARG_CHECK(stride, <=);
  auto is_dilation_nonpos = MKLDNN_CONV_ARG_CHECK(dilation, <=);
#undef MKLDNN_CONV_ARG_CHECK
  TORCH_CHECK(!is_padding_neg, "negative padding is not supported");
  TORCH_CHECK(!is_stride_nonpos, "non-positive stride is not supported");
  TORCH_CHECK(!is_dilation_nonpos, "non-positive dilation is not supported");
  TORCH_CHECK(groups > 0, "non-positive groups is not supported");

  int64_t k = input.ndimension();
  const IntArrayRef& weight_sizes = weight.sizes();
  int64_t weight_dim = weight_sizes.size();

  TORCH_CHECK(weight_dim == k,
              "Expected ", weight_dim, "-dimensional input for ", weight_dim,
              "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
              input.sizes(), " instead");
  TORCH_CHECK(weight_sizes[0] >= groups,
              "Given groups=", groups, ", expected weight to be at least ", groups,
              " at dimension 0, but got weight of size ", weight_sizes, " instead");
  TORCH_CHECK(weight_sizes[0] % groups == 0,
              "Given groups=", groups, ", expected weight to be divisible by ",
              groups, " at dimension 0, but got weight of size [", weight_sizes,
              "] instead");
  TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
              "Given groups=", groups, ", weight of size ", weight_sizes,
              ", expected input", input.sizes(), " to have ",
              (weight_sizes[1] * groups), " channels, but got ", input.size(1),
              " channels instead");
  TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
              "Given weight of size ", weight_sizes,
              ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
              ", but got bias of size ", bias.sizes(), " instead");

  std::vector<int64_t> input_shape;
  std::vector<int64_t> kernel_shape;
  bool kernel_size_correct = true;

  for (const auto i : c10::irange(2, k)) {
    input_shape.push_back(input.size(i) + 2 * padding[i-2]);
    // log new kernel size considering dilation
    kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
    if (input_shape.back() < kernel_shape.back()) {
      kernel_size_correct = false;
    }
  }

  TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

  if (!kernel_size_correct) {
    // If kernel size is incorrect
    std::ostringstream input_ss;
    std::ostringstream kernel_ss;
    std::string separator = "";

    for (int i = 0, len = input_shape.size(); i < len; ++i) {
      input_ss << separator << input_shape[i];
      kernel_ss << separator << kernel_shape[i];
      separator = " x ";
    }

    TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). "
                "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
  }
}

#define MKLDNNTensor(itensor, options)                                  \
  new_with_itensor_mkldnn(                                              \
      std::move(itensor),                                               \
      optTypeMetaToScalarType(options.dtype_opt()),                     \
      options.device_opt())

// Note [MKLDNN Convolution Memory Formats]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MKLDNN has 3 types of memory formats in convolution:
//
// In case memory format passed from PyTorch (aka. user layout)
// differs from the internal layout which MKLDNN used, a `reorder` is needed;
// otherwise when user layout is identical to internal layout,
// MKLDNN uses a memory `view` upon an existing CPU tensor.
//
// 1. NCHW (CPU tensor, contiguous)
//  input reorder:  NCHW(user) -> Blocked(internal)
//  weight reorder: OIHW(user) -> Blocked(internal)
//  output reorder: Blocked(internal) -> NCHW(user)
//
// 2. NHWC: (CPU tensor, channels last)
//  input view:     NHWC(user) -> NHWC(internal)
//  weight reorder: OHWI(user) -> Blocked(internal)
//  output view:    NHWC(internal) -> NHWC(user)
//
// 3. Blocked (MKLDNN tensor):
//  By explicitly converting a tensor to mkldnn, e.g. `x.to_mkldnn()`,
//  blocked format will propagate between layers. Input, output will be in blocked format.
//
//  For inference case, weight can be prepacked into blocked format by
//  (so as to save weight reoder overhead):
//      model = torch.utils.mkldnn.to_mkldnn(model)
//
//  For training case, grad_output can be CPU tensor or MKLDNN tensor,
//  but weight/bias and grad_weight/grad_bias are always CPU tensor.
//

static inline at::MemoryFormat mkldnn_convolution_memory_format(int64_t dims, bool is_channels_last) {
   auto memory_format =  at::MemoryFormat::Contiguous;
   if (is_channels_last) {
      memory_format = dims == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
   }
   return memory_format;
}

void _mkldnn_convolution_out (
    const Tensor& input_t,
    const Tensor& weight_t,
    const Tensor& bias,
    std::vector<int64_t>& output_sizes,
    ideep::tensor& y,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef padding,
    int64_t groups,
    bool is_channels_last,
    const ideep::attr_t& op_attr) {
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);
  auto input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  auto weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  const ideep::tensor x = itensor_from_tensor(input);
  const ideep::tensor w = itensor_from_tensor(weight);
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::convolution_forward::compute_v3(
        x,
        w,
        b,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        is_channels_last,
        op_attr);
  } else {
    ideep::convolution_forward::compute_v3(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        is_channels_last,
        op_attr);
  }
}

Tensor _mkldnn_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr = "none",
    torch::List<c10::optional<at::Scalar>> scalars =
        torch::List<c10::optional<at::Scalar>>(),
    c10::optional<c10::string_view> algorithm = c10::nullopt) {
  ideep::attr_t op_attr = ideep::attr_t();
  if (attr != "none") {
    auto it = fx_fusion_attr_map().find(attr);
    TORCH_CHECK(it != fx_fusion_attr_map().end(), "Fusion behavior undefined.");
    op_attr = it->second(scalars, algorithm);
  }
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  if (input_t.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_convolution: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  check_shape_forward(input_t, weight_t, bias, padding, stride, dilation, groups);

  bool is_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);

  auto output_sizes = conv_output_size(input_t.sizes(), weight_t.sizes(), padding, stride, dilation);
  auto output = at::empty({0}, input_t.options());
  ideep::tensor y;
  if (is_channels_last) {
    output.resize_(output_sizes, memory_format);
    y = itensor_from_tensor(output);
  }
  _mkldnn_convolution_out(
    input_t, weight_t, bias, output_sizes, y, stride, dilation, padding, groups, is_channels_last, op_attr);

  if (input_t.is_mkldnn()) {
    return MKLDNNTensor(y, input_t.options());
  } else if (!is_channels_last) {
    return mkldnn_to_dense(MKLDNNTensor(y, input_t.options()));
  } else {
    TORCH_INTERNAL_ASSERT(y.get_desc().is_nhwc());
    return output;
  }
}

Tensor mkldnn_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  return _mkldnn_convolution(
      input_t, weight_t, bias_opt, padding, stride, dilation, groups);
}

Tensor mkldnn_convolution_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm) {
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  return _mkldnn_convolution(
      input_t,
      weight_t,
      bias_opt,
      padding,
      stride,
      dilation,
      groups,
      attr,
      scalars,
      algorithm);
}

Tensor mkldnn_convolution_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    c10::string_view attr) {
  TORCH_CHECK(
      input_t.ndimension() == 4 || input_t.ndimension() == 5,
      "mkldnn_convolution_pointwise_binary: currently only support 2d and 3d")

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // Make sure inputs have same type(device, layout, dtype), device is cpu and
  // dtype is float or bfloat16.
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);

  check_shape_forward(
      input_t, weight_t, bias, padding, stride, dilation, groups);

  auto output_sizes = conv_output_size(
      input_t.sizes(), weight_t.sizes(), padding, stride, dilation);
  // TODO: support broadcast binary fusion.
  TORCH_CHECK(
      output_sizes == other_t.sizes(),
      "Binary Fusion's inputs should have same shape");
  // Only calling fusion path for channels_last path.
  // TODO: OneDNN doesn't optimize well for groups > 1 case, it will be enabled
  // at next OneDNN release.
  bool can_be_fused =
      groups == 1 && mkldnn_conv_use_channels_last(input_t, weight_t);

  auto it_binary = fusion_binary_alg_map().find(attr);
  TORCH_CHECK(
      it_binary != fusion_binary_alg_map().end(), "Fusion behavior undefined.");
  if (can_be_fused) {
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    auto memory_format =
        mkldnn_convolution_memory_format(input_t.ndimension(), true);
    auto input = input_t.contiguous(memory_format);
    auto weight = weight_t.contiguous(memory_format);
    auto other = other_t.contiguous(memory_format);
    auto output = at::empty_like(other);
    const ideep::tensor x = itensor_from_tensor(input);
    const ideep::tensor w = itensor_from_tensor(weight);
    const ideep::tensor z = itensor_from_tensor(other);
    ideep::tensor y = itensor_from_tensor(output);
    auto output_size = other.sizes().vec();
    ideep::tag format_tag = ideep::tag::nhwc;
    if (input_t.ndimension() == 5) {
      format_tag = ideep::tag::ndhwc;
    }
    auto other_desc = ideep::tensor::desc(
        output_size, get_mkldnn_dtype(weight.scalar_type()), format_tag);
    auto op_attr = ideep::attr_t::fuse_binary(it_binary->second, other_desc);
    if (bias.defined()) {
      const ideep::tensor b = itensor_from_tensor(bias);
      ideep::convolution_forward::compute_binary(
          x,
          z,
          w,
          b,
          output_size,
          y,
          {stride.begin(), stride.end()},
          {dilation.begin(), dilation.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          groups,
          /* is_channels_last */ true,
          op_attr);
    } else {
      ideep::convolution_forward::compute_binary(
          x,
          z,
          w,
          output_size,
          y,
          {stride.begin(), stride.end()},
          {dilation.begin(), dilation.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          groups,
          /* is_channels_last */ true,
          op_attr);
    }
    return output;
  } else {
    // Fallback case, if inputs are not channels last or have different dtype,
    // OneDNN fusion may have performance regression.
    Tensor output;
    if (input_t.ndimension() == 4) {
      output = at::conv2d(
          input_t, weight_t, bias_opt, stride, padding, dilation, groups);
    } else {
      output = at::conv3d(
          input_t, weight_t, bias_opt, stride, padding, dilation, groups);
    }
    if (attr == "add") {
      output.add_(other_t);
    } else if (attr == "sub") {
      output.sub_(other_t);
    } else if (attr == "mul") {
      output.mul_(other_t);
    } else {
      output.div_(other_t);
    }
    return output;
  }
}

Tensor& mkldnn_convolution_add_(
    Tensor& other_t,
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  // other_t += convolution(...)
  TORCH_CHECK(
      input_t.ndimension() == 4 || input_t.ndimension() == 5,
      "mkldnn_convolution_add_: currently only support 2d and 3d")

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // Make sure inputs have same type(device, layout, dtype), device is cpu and
  // dtype is float or bfloat16.
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);

  check_shape_forward(
      input_t, weight_t, bias, padding, stride, dilation, groups);

  auto output_sizes = conv_output_size(
      input_t.sizes(), weight_t.sizes(), padding, stride, dilation);
  TORCH_CHECK(
      output_sizes == other_t.sizes(),
      "Add Fusion's inputs should have same shape");
  // Only calling fusion path for channels_last path and the output is contiguous tensor(channels_last).
  bool can_be_fused = mkldnn_conv_use_channels_last(input_t, weight_t)
                      && (other_t.is_contiguous(at::MemoryFormat::ChannelsLast)
                          || other_t.is_contiguous(at::MemoryFormat::ChannelsLast3d));
  if (can_be_fused) {
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    ideep::tensor y = itensor_from_tensor(other_t);
    auto op_attr = ideep::attr_t::fuse_sum();
    _mkldnn_convolution_out(
      input_t, weight_t, bias, output_sizes, y, stride, dilation, padding, groups, true, op_attr);
  } else {
    // Fallback case, if inputs are not channels last or have different dtype,
    // OneDNN fusion may have performance regression.
    Tensor output;
    if (input_t.ndimension() == 4) {
      output = at::conv2d(
          input_t, weight_t, bias_opt, stride, padding, dilation, groups);
    } else {
      output = at::conv3d(
          input_t, weight_t, bias_opt, stride, padding, dilation, groups);
    }
    other_t.add_(output);
  }
  return other_t;
}

Tensor& mkldnn_convolution_add_relu_(
    Tensor& other_t,
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  // other_t += convolution(...) and then do other_t.relu_().
  TORCH_CHECK(
      input_t.ndimension() == 4 || input_t.ndimension() == 5,
      "mkldnn_convolution_add_relu_: currently only support 2d and 3d")

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  // Make sure inputs have same type(device, layout, dtype), device is cpu and
  // dtype is float or bfloat16.
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);
  check_shape_forward(
      input_t, weight_t, bias, padding, stride, dilation, groups);

  auto output_sizes = conv_output_size(
      input_t.sizes(), weight_t.sizes(), padding, stride, dilation);
  TORCH_CHECK(
      output_sizes == other_t.sizes(),
      "Add_ReLU Fusion's inputs should have same shape");
  // Only calling fusion path for channels_last path and the output is contiguous tensor(channels_last).
  bool can_be_fused = mkldnn_conv_use_channels_last(input_t, weight_t)
                      && (other_t.is_contiguous(at::MemoryFormat::ChannelsLast)
                          || other_t.is_contiguous(at::MemoryFormat::ChannelsLast3d));
  if (can_be_fused) {
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    ideep::tensor y = itensor_from_tensor(other_t);
    auto op_attr = ideep::attr_t::residual();
    _mkldnn_convolution_out(
      input_t, weight_t, bias, output_sizes, y, stride, dilation, padding, groups, true, op_attr);
    return other_t;
  } else {
    // Fallback case, if inputs are not channels last or have different dtype,
    // OneDNN fusion may have performance regression.
    Tensor output;
    if (input_t.ndimension() == 4) {
      output = at::conv2d(
          input_t, weight_t, bias_opt, stride, padding, dilation, groups);
    } else {
      output = at::conv3d(
          input_t, weight_t, bias_opt, stride, padding, dilation, groups);
    }
    return at::native::add_relu_(other_t, output);
  }
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last) {
  auto grad_input = at::empty({0}, grad_output.options());

  auto grad_y = itensor_from_tensor(grad_output);
  auto w = itensor_view_from_dense(weight);

  ideep::tensor grad_x;
  if (is_channels_last) {
    auto memory_format = mkldnn_convolution_memory_format(grad_output.ndimension(), is_channels_last);
    grad_input.resize_(input_size, memory_format);
    grad_x = itensor_from_tensor(grad_input);
  }
  ideep::convolution_backward_data::compute_v2(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups,
      is_channels_last);

  if (grad_output.is_mkldnn()) {
    return MKLDNNTensor(grad_x, grad_output.options());
  } else if (!is_channels_last){
    return mkldnn_to_dense(MKLDNNTensor(grad_x, grad_output.options()));
  } else {
    TORCH_INTERNAL_ASSERT(grad_x.get_desc().is_nhwc());
    return grad_input;
  }
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool is_channels_last) {
  const ideep::tensor grad_y = itensor_from_tensor(grad_output);
  const ideep::tensor x = itensor_from_tensor(input);

  ideep::tensor grad_w, grad_b;
  if (bias_defined) {
    ideep::convolution_backward_weights::compute_v2(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        grad_b,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        is_channels_last);
  } else {
    ideep::convolution_backward_weights::compute_v2(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        is_channels_last);
  }

  if (!is_channels_last) {
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  } else {
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())).to(at::MemoryFormat::ChannelsLast),
        bias_defined ? mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())) : Tensor());
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  bool is_channels_last = mkldnn_conv_use_channels_last(input_t, weight_t);
  auto memory_format = mkldnn_convolution_memory_format(input_t.ndimension(), is_channels_last);
  Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous(memory_format);

  Tensor input = input_t.is_mkldnn() ? input_t : input_t.contiguous(memory_format);
  Tensor weight = weight_t.is_mkldnn() ? weight_t : weight_t.contiguous(memory_format);
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2], is_channels_last);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2], is_channels_last);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

REGISTER_ALL_CPU_DISPATCH(mkldnn_convolution_backward_stub, &mkldnn_convolution_backward);

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise"),
      TORCH_FN(mkldnn_convolution_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_pointwise.binary"),
      TORCH_FN(mkldnn_convolution_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_add_"),
      TORCH_FN(mkldnn_convolution_add_));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_convolution_add_relu_"),
      TORCH_FN(mkldnn_convolution_add_relu_));
}

}}  // namespace at::native

#endif
