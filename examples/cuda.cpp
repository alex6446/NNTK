#include "NNTK/MX/Array/Predefinition.hpp"
#include <vector>
#include <iostream>
#include <any>

#define NN_GPU_ENABLED
#include "NNTK/MX/Array.hpp"

void
test_speed()
{
  //NN::MX::Array<float> a = NN::MX::Array<float>::random({100000000}, 100, 10000);
  //NN::MX::Array<float> b = NN::MX::Array<float>::random({100000000}, 100, 10000);


  auto a = NN::MX::Array<float, NN::GPU>::random({90000001}, 100, 10000);
  auto b = NN::MX::Array<float, NN::GPU>::random({90000001}, 100, 10000);
  clock_t begin = clock();
  srand(time(NULL));

  std::cout << "START DOING STUFF" << std::endl;
  //auto c = a + b + b + b + b + b + b + b; // too much
  auto c = a - b * 7; // too much

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  std::cout << "TIME: " << elapsed_secs << "s" << std::endl;



  //auto at = NN::MX::Array<float, NN::CPU>::random({90000001}, 100, 10000);
  //auto bt = NN::MX::Array<float, NN::CPU>::random({90000001}, 100, 10000);
  NN::MX::Array<float, NN::CPU> at = a;
  NN::MX::Array<float, NN::CPU> bt = b;
  begin = clock();

  std::cout << "START DOING STUFF" << std::endl;
  //auto ct = at + bt + bt + bt + bt + bt + bt + bt; // too much
  auto ct = at - bt * 7; // too much

  end = clock();
  elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  std::cout << "TIME: " << elapsed_secs << "s" << std::endl;

  std::cout << "Are equal: " << (c == ct ? "equal" : "not") << std::endl;

  std::cout << a.device() << std::endl;
  std::cout << b.device() << std::endl;
  std::cout << c.device() << std::endl;

  //int device = -1;
  //cudaGetDevice(&device);
  //cudaMemPrefetchAsync(a.data(), sizeof(NN::nn_type)*a.size(), device, NULL);
  //NN::MX::Array<float> result;
  //result = a + b;
  //result = a - b;
}

void
test_accuracy()
{

  auto a = NN::MX::Array<int, NN::GPU>::random({10}, 0, 10);
  auto b = NN::MX::Array<int, NN::GPU>::random({10}, 0, 10);
  auto c = NN::MX::Array<int, NN::GPU>::random({10}, 0, 10);
  auto d = NN::MX::Array<int, NN::GPU>::random({10}, 0, 10);
  std::cout << a << " " << &a << " " << a.device() << std::endl;
  std::cout << b << " " << &b << " " << b.device() << std::endl;
  std::cout << c << " " << &c << " " << c.device() << std::endl;
  std::cout << d << " " << &d << " " << d.device() << std::endl;
  //std::cout << d << " " << d.device() << std::endl;
  std::cout << "START DOING STUFF\n" << std::endl;
  //auto e = a - 3 - b + c + d;
  auto e = a - (b - 7); // too much
  std::cout << "\nEND DOING STUFF" << std::endl;
  e(0) = -3;
  std::cout << a << " " << a.device() << std::endl;
  std::cout << b << " " << b.device() << std::endl;
  std::cout << c << " " << c.device() << std::endl;
  std::cout << d << " " << d.device() << std::endl;
  std::cout << e << " " << e.device() << std::endl;

}

int
main()
{
  std::vector<int> a;

  test_speed();
  //test_accuracy();
}
