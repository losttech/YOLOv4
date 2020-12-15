namespace tensorflow {
    using System;
    using System.Collections;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;

    using Python.Runtime;

    class BufferedEnumerable<T> : IEnumerable<T>, ICollection {
        readonly IReadOnlyList<T> lazyList;
        readonly int bufferSize;

        public BufferedEnumerable(IReadOnlyList<T> lazyList, int bufferSize) {
            this.lazyList = lazyList ?? throw new ArgumentNullException(nameof(lazyList));
            if (bufferSize < 1) throw new ArgumentOutOfRangeException(nameof(bufferSize));
            this.bufferSize = bufferSize;
        }

        public int Count => this.lazyList.Count;

        public IEnumerator<T> GetEnumerator() {
            var buffer = new BlockingCollection<Task<T>>(boundedCapacity: this.bufferSize);
            var readyToRun = new BlockingCollection<Task<T>>(boundedCapacity: this.bufferSize);

            void Load() {
                while (!readyToRun.IsCompleted) {
                    var task = readyToRun.Take();
                    buffer.Add(task);
                }
                buffer.CompleteAdding();
            }

            void QueueLoading() {
                for(int i = 0; i < this.lazyList.Count; i++) {
                    int index = i;
                    var task = new Task<T>(() => this.lazyList[index]);
                    readyToRun.Add(task);
                    task.Start();
                }
                readyToRun.CompleteAdding();
            }

            Task.Run(QueueLoading);
            Task.Run(Load);

            return buffer.GetConsumingEnumerable()
                .Select(t => {
                    IntPtr multithreadHandle = PythonEngine.BeginAllowThreads();
                    try {
                        return t.Result;
                    } finally {
                        PythonEngine.EndAllowThreads(multithreadHandle);
                    }
                })
                .GetEnumerator();
        }
        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();

        bool ICollection.IsSynchronized => false;
        object ICollection.SyncRoot => this.lazyList;
        void ICollection.CopyTo(Array array, int index) => throw new NotImplementedException();
    }
}
