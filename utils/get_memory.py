import psutil

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used = memory_info.rss / (1024 * 1024)  # Convert bytes to megabytes
    print(f"Memory used: {memory_used:.2f} MB")