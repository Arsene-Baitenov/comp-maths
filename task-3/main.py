from pathlib import Path
from pathos.pools import ProcessPool
from compression import *


def read_all_bytes(filepath):
    with open(filepath, 'rb') as f:
        return f.read()


def write_all_bytes(filepath, data):
    with open(filepath, 'wb') as f:
        f.write(data)


def process_file(ratio, bmppath, outbmpdir):
    np_compressor = WithNumpySvdCompressor()

    start_time = time.time() * 1000
    tmp = np_compressor.to_svd(bmppath, ratio)
    end_time = time.time() * 1000

    new_image = np_compressor.from_svd(tmp)
    new_image.save(os.path.join(outbmpdir, f'numpy-{ratio}.BMP'))

    pow_compressor = PowerSvdCompressor(end_time - start_time)
    tmp = pow_compressor.to_svd(bmppath, ratio)

    new_image = pow_compressor.from_svd(tmp)
    new_image.save(os.path.join(outbmpdir, f'power-{ratio}.BMP'))

    bpow_compressor = BlockPowerSvdCompressor(end_time - start_time)
    tmp = bpow_compressor.to_svd(bmppath, ratio)

    new_image = bpow_compressor.from_svd(tmp)
    new_image.save(os.path.join(outbmpdir, f'bpower-{ratio}.BMP'))


currdir = os.getcwd()

dirs = [d for d in Path(os.path.join(currdir, 'source_bmp')).iterdir() if d.is_dir()]
src_bmp = []
out_bmp_dirs = []
for dir in dirs:
    for f in Path(dir).iterdir():
        if f.is_file() and f.suffix == '.bmp':
            src_bmp.append(f.__str__())
            out_bmp_dirs.append(os.path.join(currdir, 'out_bmp', Path(dir).name, Path(f).stem))

for (src_bmp, out_bmp_dir) in zip(src_bmp, out_bmp_dirs):
    process_file(5, src_bmp, out_bmp_dir)
