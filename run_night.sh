cd ~/Project/Lenta-Tech-Life-Hack-2026

# Сохрани в run_day.sh
chmod +x run_day.sh

# Проверь что poetry в PATH сейчас
which poetry   # должно показать путь

# Запусти
nohup ./run_day.sh > /dev/null 2>&1 &
RUNNER_PID=$!
disown $RUNNER_PID
echo "Runner PID: $RUNNER_PID"

# Через 30 секунд проверь что он реально начал
sleep 30
tail -20 reports/day_run.log