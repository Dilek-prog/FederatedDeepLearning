import docker
import time
import os

client = docker.from_env()
shared_volume_path = os.path.abspath("shared-data")

# Stelle sicher, dass das shared volume Verzeichnis existiert
os.makedirs(shared_volume_path, exist_ok=True)

def wait_for_file(path, timeout=30):
    for _ in range(timeout):
        if os.path.exists(path):
            return True
        time.sleep(1)
    return False

def run_container(name, image, volumes, **kwargs):
    print(f"🟢 Starte {name} ...")
    container = client.containers.run(
        image=image,
        name=name,
        detach=True,
        remove=True,
        volumes=volumes,
        **kwargs
    )
    container.wait()  # Blockiert bis fertig
    print(f"✅ {name} beendet")
    return container

def main():
    # Build Images (nur beim ersten Mal nötig)
    print("🔨 Baue Images ...")
    client.images.build(path="./FederatedDeepLeaning", tag="FederatedDeepLeaning:latest")

    volumes = {
        shared_volume_path: {'bind': '/shared-data', 'mode': 'rw'}
    }

    # Container 1
    run_container("FederatedDeepLeaning", "FederatedDeepLeaning:latest", volumes)

    # Warte auf done.flag
    if not wait_for_file(f"{shared_volume_path}/done.flag"):
        print("Warten auf done.flag")
        return

    # Container 2
    run_container("FederatedDeepLeaning", "FederatedDeepLeaning:latest", volumes)

    # Ergebnis anzeigen
    with open(f"{shared_volume_path}/final_output.txt", "r") as f:
        print("📄 Ergebnis aus Container2:")
        print(f.read())

    # Optional: Aufräumen
    os.remove(f"{shared_volume_path}/done.flag")
    os.remove(f"{shared_volume_path}/output.txt")
    os.remove(f"{shared_volume_path}/final_output.txt")

if __name__ == "__main__":
    main()
