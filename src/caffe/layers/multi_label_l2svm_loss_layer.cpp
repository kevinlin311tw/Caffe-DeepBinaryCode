#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelL2SVMLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  //LOG(ERROR) << bottom[0]->num() << " " << bottom[0]->channels() << " " << bottom[0]->height() << " " << bottom[0]->width() ;
  //LOG(ERROR) << bottom[1]->num() << " " << bottom[1]->channels() << " " << bottom[1]->height() << " " << bottom[1]->width() ;
 
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  if (top.size() >= 1) {
   // sigmoid cross entropy loss (averaged across batch)
    top[0]->Reshape(1, 1, 1, 1);
  }
  if (top.size() == 2) {
   // softmax output
    top[1]->ReshapeLike(*sigmoid_output_.get());
    top[1]->ShareData(*sigmoid_output_.get());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void MultiLabelL2SVMLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  Dtype forward_temp = 0;
  Dtype total_loss = 0;
  for (int i = 0; i < count; ++i) {
    if (target[i] != 0) {    	// Update the loss only if target[i] is not 0
	//loss -= input_data[i] * ((target[i] > 0) - (input_data[i] >= 0));

	if((target[i] > 0)&&(input_data[i] >= 1))
	{
      		loss = 0;
		forward_temp = 0;
	}
	else if((target[i] > 0)&&(input_data[i] <1))
	{
      		loss = (1-input_data[i])*(1-input_data[i]);
		forward_temp = (1-input_data[i]);
	}
	else if((target[i] < 0)&&(input_data[i] <= -1))
	{
      		loss = 0;
		forward_temp = 0;
	}
	else if ((target[i] < 0)&&(input_data[i] > -1))
	{
     		loss =  (input_data[i]+1)*(input_data[i]+1);
		forward_temp = input_data[i]+1;
	}

	diff_.mutable_cpu_data()[i] = forward_temp;//buffer
	total_loss = total_loss + loss;
    }
  }
    if (top.size() >= 1) {
      top[0]->mutable_cpu_data()[0] = total_loss / num / Dtype(2);
    }
  
  //return total_loss/num/Dtype(2) ;

}

template <typename Dtype>
void MultiLabelL2SVMLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* output_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      if (target[i] != 0) {
	if (target[i] > 0)
	{
		if (output_data[i]<1){	
		bottom_diff[i] = Dtype(-1)*diff_.cpu_data()[i];
		}else{
		bottom_diff[i] = 0;
		}
	}
	else if(target[i]<0)
	{
		if (output_data[i]>-1){
		bottom_diff[i] = Dtype(1)*diff_.cpu_data()[i];
		}else{
		bottom_diff[i] = 0;
		}
	}
      } else {
        bottom_diff[i] = 0;
      }
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
//STUB_GPU_BACKWARD(MultiLabelL2SVMLossLayer, Backward);
STUB_GPU(MultiLabelL2SVMLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelL2SVMLossLayer);
REGISTER_LAYER_CLASS(MultiLabelL2SVMLoss);

}  // namespace caffe
