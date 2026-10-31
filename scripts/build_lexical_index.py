"""Build a local full-catalog Chinese BM25 index without query/click labels."""
import argparse
from pathlib import Path
import pyarrow as pa
from recall.lexical import DEFAULT_INDEX, build_index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes',type=Path,default=Path('datasets/notes'))
    parser.add_argument('--output',type=Path,default=DEFAULT_INDEX)
    parser.add_argument('--title-chars',type=int,default=128)
    parser.add_argument('--body-chars',type=int,default=512)
    parser.add_argument('--title-weight',type=float,default=2.)
    parser.add_argument('--body-weight',type=float,default=1.)
    parser.add_argument('--query-chars',type=int,default=128)
    parser.add_argument('--batch-size',type=int,default=8192)
    parser.add_argument('--cache-mib',type=int,default=256)
    parser.add_argument('--memory-limit-mib',type=int,default=2048)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    pa.set_cpu_count(args.threads);pa.set_io_thread_count(args.threads)
    build_index(args.notes,args.output,title_chars=args.title_chars,body_chars=args.body_chars,
                title_weight=args.title_weight,body_weight=args.body_weight,query_chars=args.query_chars,
                batch_size=args.batch_size,cache_mib=args.cache_mib,memory_limit_mib=args.memory_limit_mib)


if __name__=='__main__':main()