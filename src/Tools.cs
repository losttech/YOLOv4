namespace tensorflow {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    static class Tools {
        public static IEnumerable<bool> Repeat(int times) => Enumerable.Repeat(true, times);
        [Obsolete("Use random.shuffle for reproducibility")]
        public static void Shuffle<T>(IList<T> list) {
            var random = new Random();
            for(int i = list.Count - 1; i > 0; i--) {
                int swapWith = random.Next(i+1);
                Swap(list, i, swapWith);
            }
        }

        public static void Swap<T>(IList<T> list, int index1, int index2) {
            T tmp = list[index1];
            list[index1] = list[index2];
            list[index2] = tmp;
        }

        public static T[] Slice<T>(this T[] array, Range range) {
            if (array is null) throw new ArgumentNullException(nameof(array));

            var (offset, len) = range.GetOffsetAndLength(array.Length);
            var result = new T[len];
            Array.Copy(array, offset, result, 0, len);
            return result;
        }

        public static string[] NonEmptyLines(string filePath)
            => System.IO.File.ReadAllLines(filePath)
                .Select(l => l.Trim())
                .Where(l => !string.IsNullOrEmpty(l))
                .ToArray();

        public static void Deconstruct<T>(this T[] array, out T i0, out T i1, out T i2, out T i3) {
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (array.Length != 4) throw new ArgumentException();

            i0 = array[0];
            i1 = array[1];
            i2 = array[2];
            i3 = array[3];
        }
        public static void Deconstruct<T>(this T[] array, out T i0, out T i1, out T i2) {
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (array.Length != 3) throw new ArgumentException();

            i0 = array[0];
            i1 = array[1];
            i2 = array[2];
        }

        public static IEnumerable<(int, T)> Enumerate<T>(params T[] items)
            => items.Select((index, item) => (item, index));

        public static IEnumerable<T> BufferedEnumerate<T>(this IReadOnlyList<T> list, int bufferSize)
            => new BufferedEnumerable<T>(list, bufferSize);
    }
}
