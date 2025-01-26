# import platform
import psutil
import cpuinfo
# import torch
import subprocess

def get_cpu_info():
    cpu_info = cpuinfo.get_cpu_info()
    try:
        cpu_freq = psutil.cpu_freq()
        frequency = f"{cpu_freq.current:.2f} MHz" if cpu_freq else "N/A"
    except Exception as e:
        frequency = "N/A"  # Handle the error gracefully

    return {
        "brand": cpu_info['brand_raw'],
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "frequency": frequency,
        "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    }

def get_gpu_info():
    try:
        # Get general GPU info
        gpu_info = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode('utf-8')
        gpu_names = [line.split(":")[1].strip() for line in gpu_info.split('\n') if "Chipset Model:" in line]

        # Get VRAM info for dedicated GPU
        vram_info = subprocess.check_output(["ioreg", "-l", "-w", "0", "-r", "-c", "IOPCIDevice"]).decode('utf-8')
        vram_lines = [line for line in vram_info.split('\n') if "VRAM,totalMB" in line]

        if vram_lines:
            vram_mb = vram_lines[0].split('=')[1].strip()
            vram_gb = int(vram_mb) / 1024
            memory_info = f"{vram_gb:.2f} GB (dedicated)"
        else:
            memory_info = "Unable to retrieve dedicated VRAM info"

        return {
            "name": " & ".join(gpu_names),
            "memory": memory_info
        }
    except Exception as e:
        return {"name": f"Unable to retrieve GPU info: {str(e)}", "memory": "N/A"}

def print_hardware_info():
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()

    print("CPU Information:")
    print(f"  Brand: {cpu_info['brand']}")
    print(f"  Cores: {cpu_info['cores']}")
    print(f"  Threads: {cpu_info['threads']}")
    print(f"  Frequency: {cpu_info['frequency']}")
    print(f"  RAM: {cpu_info['ram']}")

    print("\nGPU Information:")
    print(f"  Name: {gpu_info['name']}")
    print(f"  Memory: {gpu_info['memory']}")

if __name__ == "__main__":
    print_hardware_info()
