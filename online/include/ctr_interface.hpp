#include <string>
#include <vector>
namespace tf_interface{
#define EXPORT __attribute__((visibility("default")))
struct EXPORT Shape{
    std::vector<int> shape; 
};
//init resouce
EXPORT bool  init(const std::string& model_path);
//process feature
EXPORT bool process(int* feature, int length, const Shape& shape, std::vector<float>& res_vec);

//release the feature
EXPORT void  release();
}
#endif
