
import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Configure Drain3 for Cisco log pattern discovery
cfg = TemplateMinerConfig()
cfg.max_depth = 4
drain = TemplateMiner(config=cfg)

def run_log_parsing():
    SUB_FOLDERS = ['logs/bgp', 'logs/CallMng', 'logs/Cisco']
    log_files = []
    for sub in SUB_FOLDERS:
        log_files.extend(glob.glob(f"{sub}/**/*.*", recursive=True))

    for path in log_files:
        print(f">>> Parsing: {path}")
        basename = os.path.basename(path)
        records = []

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Use Drain3 to extract templates and variables
                result = drain.add_log_message(line)
                records.append({
                    "raw_text": line,
                    "template": result.get("template_mined", ""),
                    "source": basename
                })

        # Save as Parquet for high-speed reading
        out_file = f"data/{basename}.parquet"
        pq.write_table(pa.Table.from_pylist(records), out_file, compression='zstd')
        print(f"  Wrote {len(records)} rows to {out_file}")

if __name__ == "__main__":
    run_log_parsing()


# # 1_parse.py

# # Entry point to load documents (pdf/txt/etc.)

# # Splits text into chunks

# # Output: structured chunks (often JSON / list of chunks)