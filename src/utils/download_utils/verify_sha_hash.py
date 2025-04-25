import hashlib

# Based on https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def verify_sha256(path: str, sha256: str) -> bool:
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    digest = h.hexdigest()
    return digest == sha256