from pathlib import Path
# Make a bulk system for nutmeg

dipep = Path("./ace-ala-nme.charges").read_text().split()
wat = Path("./water.charges").read_text().split()
for _ in range(630):
    dipep.extend(wat)
Path("./ala-dipep-bulk.charges").write_text("\n".join(dipep))
