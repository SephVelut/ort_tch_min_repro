pip deps use CUDA wheel not CPU

python3.10 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
cargo run

Keep running cargo run its a race condition

That should end with:

malloc(): largebin double linked list corrupted (bk)
