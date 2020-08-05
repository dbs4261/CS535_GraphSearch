//
// Created by Daniel Simon on 8/5/20.
//

#include <numeric>

#include "coloring.hpp"

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