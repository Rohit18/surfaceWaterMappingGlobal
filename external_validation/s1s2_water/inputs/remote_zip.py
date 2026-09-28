#!/usr/bin/env python3
"""List or selectively extract members from a remote ZIP using HTTP ranges.

Restored from the user's pasted helper; validates ranges and extraction paths.
"""
from __future__ import annotations
import argparse
import fnmatch
import io
import os
import time
import urllib.error
import urllib.request
import zipfile

class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        with self._open(urllib.request.Request(url, method='HEAD')) as response:
            self.size = int(response.headers['Content-Length'])
        self.position = 0

    @staticmethod
    def _open(request):
        for attempt in range(6):
            try:
                return urllib.request.urlopen(request, timeout=120)
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 5:
                    raise
                time.sleep(3 * (attempt + 1))

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f'Unsupported whence: {whence}')
        if position < 0:
            raise ValueError('Negative seek position')
        self.position = position
        return position

    def read(self, size=-1):
        if self.position >= self.size:
            return b''
        end = self.size - 1 if size is None or size < 0 else min(self.position + size, self.size) - 1
        if end < self.position:
            return b''
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{end}'})
        with self._open(request) as response:
            expected = f'bytes {self.position}-{end}/{self.size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise RuntimeError('Server did not honor the requested byte range')
            data = response.read()
        if len(data) != end - self.position + 1:
            raise IOError('Incomplete range response')
        self.position += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--match', action='append', default=[])
    parser.add_argument('--extract-to')
    args = parser.parse_args()
    with HTTPRangeReader(args.url) as remote, zipfile.ZipFile(remote) as archive:
        patterns = args.match or ['*']
        for info in archive.infolist():
            if not any(fnmatch.fnmatch(info.filename, pattern) for pattern in patterns):
                continue
            print(f'{info.file_size}\t{info.compress_size}\t{info.filename}', flush=True)
            if args.extract_to and not info.is_dir():
                root = os.path.realpath(args.extract_to)
                destination = os.path.realpath(os.path.join(root, info.filename))
                if os.path.commonpath([root, destination]) != root:
                    raise ValueError('ZIP member escapes destination')
                if os.path.exists(destination) and os.path.getsize(destination) == info.file_size:
                    print(f'SKIP complete\t{destination}', flush=True)
                    continue
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                temporary = destination + '.part'
                with archive.open(info) as source, open(temporary, 'wb') as target:
                    while chunk := source.read(64 * 1024 * 1024):
                        target.write(chunk)
                os.replace(temporary, destination)

if __name__ == '__main__':
    main()
