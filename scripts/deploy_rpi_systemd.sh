#!/usr/bin/env bash
set -euo pipefail

# Prosty skrypt "fire and forget" do wdrożenia pipeline'u na Raspberry Pi / Ubuntu.
# Zakłada, że uruchamiasz go z katalogu repozytorium jako root (lub przez sudo):
#   sudo bash scripts/deploy_rpi_systemd.sh

PROJECT_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="/opt/vision-pipeline"
PYTHON_BIN="/usr/bin/python3"   # w razie potrzeby zmień na inny interpreter

SYSTEMD_DIR_SRC="${PROJECT_SRC_DIR}/config/systemd"
SYSTEMD_DIR_DST="/etc/systemd/system"

echo "[1/4] Kopiuję projekt do ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"
rsync -a --delete "${PROJECT_SRC_DIR}/" "${TARGET_DIR}/"

echo "[2/4] (Opcjonalnie) Tworzę środowisko wirtualne i instaluję zależności..."
if command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

cd "${TARGET_DIR}"
if [ ! -d .venv ]; then
  echo "  - Tworzenie .venv przy użyciu ${PY}"
  "${PY}" -m venv .venv || echo "  ! Nie udało się utworzyć venv – możesz użyć systemowego Pythona"
fi

if [ -d .venv ]; then
  echo "  - Instalacja zależności w .venv"
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "  ! Brak .venv – upewnij się, że wymagane pakiety są zainstalowane globalnie"
fi

echo "[3/4] Instaluję jednostki systemd i timery..."
cp "${SYSTEMD_DIR_SRC}/day_collector.service"       "${SYSTEMD_DIR_DST}/"
cp "${SYSTEMD_DIR_SRC}/day_collector_start.timer"  "${SYSTEMD_DIR_DST}/"
cp "${SYSTEMD_DIR_SRC}/day_collector_stop.service" "${SYSTEMD_DIR_DST}/"
cp "${SYSTEMD_DIR_SRC}/day_collector_stop.timer"   "${SYSTEMD_DIR_DST}/"
cp "${SYSTEMD_DIR_SRC}/night_trainer.service"      "${SYSTEMD_DIR_DST}/"
cp "${SYSTEMD_DIR_SRC}/train_classifier.service"   "${SYSTEMD_DIR_DST}/"

# Upewnij się, że pliki service wskazują na właściwy interpreter Pythona i katalog projektu.
# Domyślnie w repo jest /usr/bin/python i /opt/vision-pipeline –
# jeżeli używasz venv, możesz zaktualizować ExecStart w plikach service
# aby korzystały z /opt/vision-pipeline/.venv/bin/python.

echo "[4/4] Przeładowuję systemd, włączam i uruchamiam timery..."
systemctl daemon-reload

systemctl enable day_collector_start.timer
systemctl enable day_collector_stop.timer

systemctl start day_collector_start.timer
systemctl start day_collector_stop.timer
echo "Timery dla Night Trainer i Train Classifier nie są już używane –"
echo "uruchamiają się sekwencyjnie dzięki zależnościom OnSuccess."

echo
echo "Gotowe. Pipeline będzie działał wg harmonogramu systemd (07:00–15:00 + nocne batch'e)."
echo "Sprawdź status timerów np.: systemctl list-timers | grep day_collector"
