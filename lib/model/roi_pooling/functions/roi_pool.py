import torch
#import cupy
#import collections
from torch.autograd import Function
from .._ext import roi_pooling
import pdb

class RoIPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois): 
        ctx.feature_size = features.size()           
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                            _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                 features, rois, output, ctx.argmax)

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        roi_pooling.roi_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                              grad_output, ctx.rois, grad_input, ctx.argmax)

        return grad_input, None

#class RoIPoolFunction(torch.autograd.Function):
#	CUDA_NUM_THREADS = 1024
#	GET_BLOCKS = staticmethod(lambda N: (N + RoIPoolFunction.CUDA_NUM_THREADS - 1) // RoIPoolFunction.CUDA_NUM_THREADS)
#	Stream = collections.namedtuple('Stream', ['ptr'])
#
#	kernel_forward = b'''
#	#define FLT_MAX 340282346638528859811704183484516925440.0f
#	typedef float Dtype;
#	#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#	extern "C"
#	__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
#		const Dtype spatial_scale, const int channels, const int height,
#		const int width, const int pooled_height, const int pooled_width,
#		const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
#	  CUDA_KERNEL_LOOP(index, nthreads) { 
#		// (n, c, ph, pw) is an element in the pooled output
#		int pw = index % pooled_width;
#		int ph = (index / pooled_width) % pooled_height;
#		int c = (index / pooled_width / pooled_height) % channels;
#		int n = index / pooled_width / pooled_height / channels;
#
#		bottom_rois += n * 5;
#		int roi_batch_ind = bottom_rois[0];
#		int roi_start_w = round(bottom_rois[1] * spatial_scale);
#		int roi_start_h = round(bottom_rois[2] * spatial_scale);
#		int roi_end_w = round(bottom_rois[3] * spatial_scale);
#		int roi_end_h = round(bottom_rois[4] * spatial_scale);
#
#		// Force malformed ROIs to be 1x1
#		int roi_width = max(roi_end_w - roi_start_w + 1, 1);
#		int roi_height = max(roi_end_h - roi_start_h + 1, 1);
#		Dtype bin_size_h = static_cast<Dtype>(roi_height)
#						   / static_cast<Dtype>(pooled_height);
#		Dtype bin_size_w = static_cast<Dtype>(roi_width)
#						   / static_cast<Dtype>(pooled_width);
#
#		int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
#											* bin_size_h));
#		int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
#											* bin_size_w));
#		int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
#										 * bin_size_h));
#		int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
#										 * bin_size_w));
#
#		// Add roi offsets and clip to input boundaries
#		hstart = min(max(hstart + roi_start_h, 0), height);
#		hend = min(max(hend + roi_start_h, 0), height);
#		wstart = min(max(wstart + roi_start_w, 0), width);
#		wend = min(max(wend + roi_start_w, 0), width);
#		bool is_empty = (hend <= hstart) || (wend <= wstart);
#
#		// Define an empty pooling region to be zero
#		Dtype maxval = is_empty ? 0 : -FLT_MAX;
#		// If nothing is pooled, argmax = -1 causes nothing to be backprop'd
#		int maxidx = -1;
#		bottom_data += (roi_batch_ind * channels + c) * height * width;
#		for (int h = hstart; h < hend; ++h) {
#		  for (int w = wstart; w < wend; ++w) {
#			int bottom_index = h * width + w;
#			if (bottom_data[bottom_index] > maxval) {
#			  maxval = bottom_data[bottom_index];
#			  maxidx = bottom_index;
#			}
#		  }
#		}
#		top_data[index] = maxval;
#		argmax_data[index] = maxidx;
#	  }
#	}
#	'''
#
#	kernel_backward = b'''
#	typedef float Dtype;
#	#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#	extern "C"
#	__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
#		const int* argmax_data, const int num_rois, const Dtype spatial_scale,
#		const int channels, const int height, const int width,
#		const int pooled_height, const int pooled_width, Dtype* bottom_diff,
#		const Dtype* bottom_rois) {
#	  CUDA_KERNEL_LOOP(index, nthreads) {
#		// (n, c, h, w) coords in bottom data
#		int w = index % width;
#		int h = (index / width) % height;
#		int c = (index / width / height) % channels;
#		int n = index / width / height / channels;
#
#		Dtype gradient = 0;
#		// Accumulate gradient over all ROIs that pooled this element
#		for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
#		  const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
#		  int roi_batch_ind = offset_bottom_rois[0];
#		  // Skip if ROI's batch index doesn't match n
#		  if (n != roi_batch_ind) {
#			continue;
#		  }
#
#		  int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
#		  int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
#		  int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
#		  int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);
#
#		  // Skip if ROI doesn't include (h, w)
#		  const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
#							   h >= roi_start_h && h <= roi_end_h);
#		  if (!in_roi) {
#			continue;
#		  }
#
#		  int offset = (roi_n * channels + c) * pooled_height * pooled_width;
#		  const Dtype* offset_top_diff = top_diff + offset;
#		  const int* offset_argmax_data = argmax_data + offset;
#
#		  // Compute feasible set of pooled units that could have pooled
#		  // this bottom unit
#
#		  // Force malformed ROIs to be 1x1
#		  int roi_width = max(roi_end_w - roi_start_w + 1, 1);
#		  int roi_height = max(roi_end_h - roi_start_h + 1, 1);
#
#		  Dtype bin_size_h = static_cast<Dtype>(roi_height)
#							 / static_cast<Dtype>(pooled_height);
#		  Dtype bin_size_w = static_cast<Dtype>(roi_width)
#							 / static_cast<Dtype>(pooled_width);
#
#		  int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
#		  int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
#		  int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
#		  int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);
#
#		  phstart = min(max(phstart, 0), pooled_height);
#		  phend = min(max(phend, 0), pooled_height);
#		  pwstart = min(max(pwstart, 0), pooled_width);
#		  pwend = min(max(pwend, 0), pooled_width);
#
#		  for (int ph = phstart; ph < phend; ++ph) {
#			for (int pw = pwstart; pw < pwend; ++pw) {
#			  if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
#				gradient += offset_top_diff[ph * pooled_width + pw];
#			  }
#			}
#		  }
#		}
#		bottom_diff[index] = gradient;
#	  }
#	}
#	'''
#	cupy_init = cupy.array([])
#	compiled_forward = cupy.cuda.compiler.compile_with_cache(kernel_forward).get_function('ROIPoolForward')
#	compiled_backward = cupy.cuda.compiler.compile_with_cache(kernel_backward).get_function('ROIPoolBackward')
#
#	def __init__(self, pooled_height, pooled_width, spatial_scale):
#		self.pooled_height = pooled_height
#		self.pooled_width = pooled_width
#		self.spatial_scale = spatial_scale
#
#	def forward(self, images, rois):
#		output = torch.cuda.FloatTensor(len(rois), images.size(1) * self.pooled_height * self.pooled_width)
#		self.argmax = torch.cuda.IntTensor(output.size()).fill_(-1)
#		self.input_size = images.size()
#		self.save_for_backward(rois)
#		RoIPoolFunction.compiled_forward(grid = (RoIPoolFunction.GET_BLOCKS(output.numel()), 1, 1), block = (RoIPoolFunction.CUDA_NUM_THREADS, 1, 1), args=[
#			output.numel(), images.data_ptr(), cupy.float32(self.spatial_scale), self.input_size[-3], self.input_size[-2], self.input_size[-1],
#			self.pooled_height, self.pooled_width, rois.data_ptr(), output.data_ptr(), self.argmax.data_ptr()
#			  ], stream=RoIPoolFunction.Stream(ptr=torch.cuda.current_stream().cuda_stream))
#		return output.view(len(rois), images.size(1), self.pooled_height, self.pooled_width)
#
#	def backward(self, grad_output):
#		rois, = self.saved_tensors
#		grad_input = torch.cuda.FloatTensor(*self.input_size).zero_()
#		RoIPoolFunction.compiled_backward(grid = (RoIPoolFunction.GET_BLOCKS(grad_input.numel()), 1, 1), block = (RoIPoolFunction.CUDA_NUM_THREADS, 1, 1), args=[
#			grad_input.numel(), grad_output.data_ptr(), self.argmax.data_ptr(), len(rois), cupy.float32(self.spatial_scale), self.input_size[-3],
#			self.input_size[-2], self.input_size[-1], self.pooled_height, self.pooled_width, grad_input.data_ptr(), rois.data_ptr()
#			  ], stream=RoIPoolFunction.Stream(ptr=torch.cuda.current_stream().cuda_stream))
#		return grad_input, None
