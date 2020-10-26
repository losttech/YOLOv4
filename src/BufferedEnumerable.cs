namespace tensorflow {
    using System;
    using System.Collections;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    class BufferedEnumerable<T>: IEnumerable<T> {
        readonly IEnumerable<T> wrapped;
        readonly int bufferSize;

        public BufferedEnumerable(IEnumerable<T> enumerable, int bufferSize) {
            this.wrapped = enumerable ?? throw new ArgumentNullException(nameof(enumerable));
            if (bufferSize < 1) throw new ArgumentOutOfRangeException(nameof(bufferSize));
            this.bufferSize = bufferSize;
        }

        public IEnumerator<T> GetEnumerator() {
            var buffer = new BlockingCollection<T>(boundedCapacity: this.bufferSize);
            var enumerator = this.wrapped.GetEnumerator();
            void PreloadNext() {
                Task.Run(() => {
                    if (enumerator.MoveNext()) {
                        buffer.Add(enumerator.Current);
                        PreloadNext();
                    } else {
                        buffer.CompleteAdding();
                    }
                });
            }
            PreloadNext();
            return buffer.GetConsumingEnumerable().GetEnumerator();
        }
        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
    }
}
