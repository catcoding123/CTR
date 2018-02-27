#include <memory>
#include <cassert>
#include <vector>
#include <mutex>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/loader.h"


#include "ctr_interface.hpp"

namespace tf_interface{
using namespace std;
using namespace tensorflow;

//set symbol hidden
#define EXPORT __attribute__((visibility("default")))

static SavedModelBundle* g_session = NULL;
//global session lock
static std::mutex sess_lock{};

//init tensorflow model
EXPORT bool init(const std::string& model_path) {
    sess_lock.lock();
    if (g_session  == NULL ) {
        g_session = new SavedModelBundle;
    }else {
        sess_lock.unlock();
        return true;
    }
    Status status; 
    status = LoadSavedModel(SessionOptions(),
                            RunOptions(),model_path,
                            {kSavedModelTagTrain},g_session);
    if (!status.ok()) {
        std::cerr<<__FILE__<<" "<<__LINE__<<": init model failed! "<<status.ToString()<<std::endl;
        g_session->session->Close();
        delete g_session;
        g_session = NULL;
        sess_lock.unlock(); 
        return false;
    }
    sess_lock.unlock();
    return true;
}

static void featureToTensor(int* feature,
                         Tensor& tensor,
                         TensorShape& shape) {
    auto inputx_mapped = tensor.tensor<int32_t, 2>();
    for (int j =0 ; j < shape.dim_size(0); ++j) {
        for (int i =0 ; i < shape.dim_size(1);  ++i) {
            inputx_mapped(j,i) = feature[i+j*shape.dim_size(1)];
        }
    }
}

EXPORT bool process(
        int* feature,
        int length, 
        const Shape& shape,
	    std::vector<float>& res_vec) {

    SavedModelBundle* bundle = g_session;
    TensorShape tensor_shape;
    for (auto& iter : shape.shape) {
        tensor_shape.AddDim(iter);
    }
    Tensor x_tensor(DT_INT32,tensor_shape);
    featureToTensor(feature,x_tensor,tensor_shape);
    vector<Tensor> out_tensor;
    vector<pair<string,tensorflow::Tensor>> input_tensors;
    input_tensors.push_back(pair<string,Tensor>("data:0",x_tensor));
    Status status = bundle->session->Run(input_tensors, {"deploy/Sigmoid:0"}, {}, &out_tensor);
    
    if (!status.ok()) {
        std::cerr<<__FILE__<<"  "<<__LINE__<<": run error! "<<status.ToString()<<endl;
        return -1;
    }
	for (int i=0; i<shape.shape[0]; i++) {
		float res = out_tensor[0].flat<float>()(i);
		res_vec.push_back(res);	
	}
	return true;
}

EXPORT void release() {
    sess_lock.lock();	
    if (g_session != NULL) {
    	g_session->session->Close();
    	delete g_session;
        g_session = NULL;
    }
    sess_lock.unlock();
}

}
