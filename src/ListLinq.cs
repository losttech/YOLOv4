namespace tensorflow {
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    static class ListLinq {
        public static IReadOnlyList<TResult> Select<TSource, TResult>(
            this IReadOnlyList<TSource> source, Func<TSource, TResult> selector)
            => new LazySelectList<TSource, TResult>(source, selector);

        class LazySelectList<TOrig, T> : IReadOnlyList<T> {
            readonly IReadOnlyList<TOrig> source;
            readonly Func<TOrig, T> selector;
            public LazySelectList(IReadOnlyList<TOrig> source, Func<TOrig, T> selector) {
                this.source = source ?? throw new ArgumentNullException(nameof(source));
                this.selector = selector ?? throw new ArgumentNullException(nameof(selector));
            }
            public T this[int index] => this.selector(this.source[index]);
            public int Count => this.source.Count;
            public IEnumerator<T> GetEnumerator()
                => Enumerable.Select(this.source, this.selector).GetEnumerator();

            IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
        }
    }
}
