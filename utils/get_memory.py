import psutil
import sys

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used = memory_info.rss / (1024 * 1024)  # Convert bytes to megabytes
    print(f"Memory used: {memory_used:.2f} MB")

def list_vars_in_use():
    # Get the current frame (local variables) and the global frame (global variables)
    current_frame = sys._getframe()
    global_frame = current_frame.f_globals

    local_vars = {}
    global_vars = {}

    # Process local variables
    for var_name, var_value in current_frame.f_locals.items():
        memory_usage = sys.getsizeof(var_value)
        local_vars[var_name] = memory_usage / (1024 * 1024)

    print("Local variables: ", local_vars)

    # Process global variables
    for var_name, var_value in global_frame.items():
        memory_usage = sys.getsizeof(var_value)
        global_vars[var_name] = memory_usage / (1024 * 1024)

    print("Global variables: ", global_vars)