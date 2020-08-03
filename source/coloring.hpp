#ifndef CS535_GRAPHSEARCH_COLORING_HPP
#define CS535_GRAPHSEARCH_COLORING_HPP

#include <array>

namespace turbo {
static constexpr std::array<float, 4> kRedVec4 = {0.13572138, 4.61539260, -42.66032258, 132.13108234};
static constexpr std::array<float, 4> kGreenVec4 = {0.09140261, 2.19418839, 4.84296658, -14.18503333};
static constexpr std::array<float, 4> kBlueVec4 = {0.10667330, 12.64194608, -60.58204836, 110.36276771};
static constexpr std::array<float, 2> kRedVec2 = {-152.94239396, 59.28637943};
static constexpr std::array<float, 2> kGreenVec2 {4.27729857, 2.82956604};
static constexpr std::array<float, 2> kBlueVec2 = {-89.90310912, 27.34824973};
}

std::array<float, 3> TurboColorFloat(float value) {
  std::array<float, 4> vec4 = {1.0f, value, value * value, value * value * value};
  std::array<float, 2> vec2 =  {vec4[2] * vec4[2], vec4[2] * vec4[3]};
  return {
      std::inner_product(vec4.begin(), vec4.end(), turbo::kRedVec4.begin(), 0.0f) +
          std::inner_product(vec2.begin(), vec2.end(), turbo::kRedVec2.begin(), 0.0f),
      std::inner_product(vec4.begin(), vec4.end(), turbo::kGreenVec4.begin(), 0.0f) +
          std::inner_product(vec2.begin(), vec2.end(), turbo::kGreenVec2.begin(), 0.0f),
      std::inner_product(vec4.begin(), vec4.end(), turbo::kBlueVec4.begin(), 0.0f) +
          std::inner_product(vec2.begin(), vec2.end(), turbo::kBlueVec2.begin(), 0.0f)
  };
}

std::array<char[8], 10> brewer_spectral {"#9e0142", "#d52e4f", "#f46d43", "#fdae61", "#fee08b", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"};

#endif //CS535_GRAPHSEARCH_COLORING_HPP
