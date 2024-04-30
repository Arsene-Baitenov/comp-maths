import os
from pathlib import Path
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
    new_image.save(os.path.join(outbmpdir, Path(bmppath).stem, f'numpy-{ratio}.BMP'))

    pow_compressor = PowerSvdCompressor(end_time - start_time)
    tmp = pow_compressor.to_svd(bmppath, ratio)

    new_image = pow_compressor.from_svd(tmp)
    new_image.save(os.path.join(outbmpdir, Path(bmppath).stem, f'power-{ratio}.BMP'))

    bpow_compressor = BlockPowerSvdCompressor(end_time - start_time)
    tmp = bpow_compressor.to_svd(bmppath, ratio)

    new_image = bpow_compressor.from_svd(tmp)
    new_image.save(os.path.join(outbmpdir, Path(bmppath).stem, f'bpower-{ratio}.BMP'))


def process_dir(ratio, bmpdir, outbmpdir):
    files = [f for f in Path(bmpdir).iterdir() if f.is_file()]
    for f in files:
        process_file(ratio, f, outbmpdir)


bmpdirnames = [
    # 'bw',
    # 'contrast',
    # 'drops',
    # 'geometry',
    # 'gradient',
    # 'lines'
]

currdir = os.getcwd()

for bmpdirname in bmpdirnames:
    process_dir(5,
                os.path.join(currdir, 'source_bmp', bmpdirname),
                os.path.join(currdir, 'out_bmp', bmpdirname))

