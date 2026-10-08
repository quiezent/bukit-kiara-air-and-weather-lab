#pragma once
#include <limits>

enum class First20Call { Invalid, Rise, Drop, Unresolved };

// Match the web dashboard's three-class argmax rule. A no-crossing winner
// and any tie both mean no directional call; a winner need not exceed 50%.
// Type/null checks and the forecast clock belong to the JSON caller.
constexpr First20Call classifyFirst20(double rise, double drop, double none,
                                     double reference, First20Call declared) {
  // Bounded comparisons reject NaN and infinity without a platform API.
  if (!(rise >= 0.0 && rise <= 1.0 && drop >= 0.0 && drop <= 1.0 &&
        none >= 0.0 && none <= 1.0 && reference >= 0.0 &&
        reference <= std::numeric_limits<double>::max())) {
    return First20Call::Invalid;
  }
  // This is the web renderer's tolerance, allowing floating-point roundoff.
  constexpr double sumTolerance = 1e-8;
  const double sum = rise + drop + none;
  const double sumError = sum - 1.0;
  if (!(sumError >= -sumTolerance && sumError <= sumTolerance)) {
    return First20Call::Invalid;
  }
  const First20Call winner = rise > drop && rise > none ? First20Call::Rise
      : drop > rise && drop > none ? First20Call::Drop
      : First20Call::Unresolved;
  if (declared != winner ||
      (winner == First20Call::Drop && reference < 20.0)) {
    return First20Call::Invalid;
  }
  return winner;
}
