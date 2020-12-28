namespace tensorflow.keras.utils {
    using System;
    using System.Collections.Generic;

    using numpy;
    class ListSequence<T> : Sequence {
        readonly IReadOnlyList<T> source;

        public ListSequence(IReadOnlyList<T> source) {
            this.source = source ?? throw new ArgumentNullException(nameof(source));
        }

        public override (ndarray, object) __getitem__(int index) => throw new NotSupportedException();
        public new T this[int index] => this.source[index];

        public override dynamic __getitem___dyn(object index) => throw new NotSupportedException();

        public override int __len__() => this.source.Count;

        public override dynamic __len___dyn() => throw new NotSupportedException();
    }

    static class ListSequenceExtensions {
        public static ListSequence<T> ToSequence<T>(this IReadOnlyList<T> source)
            => new ListSequence<T>(source);
    }
}
