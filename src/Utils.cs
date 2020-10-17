namespace tensorflow.keras {
    using System.Collections.Generic;
    using System.Globalization;
    using System.Linq;

    using numpy;
    static class Utils {
        public static ndarray<float> ParseAnchors(string anchors)
            => anchors.Split(',')
                .Select(coord => float.Parse(coord.Trim(), CultureInfo.InvariantCulture))
                .ToNumPyArray()
                .reshape(new[] { 3, 3, 2 })
                .AsArray<float>();
        public static ndarray<float> ParseAnchors(IEnumerable<int> anchors)
            => anchors.ToNumPyArray()
                .reshape(new[] { 3, 3, 2 })
                .AsArray<int>()
                .AsType<float>();
    }
}
