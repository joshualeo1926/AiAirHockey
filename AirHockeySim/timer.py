import time

class Timer():
    def __init__(self):
        self.start = datetime.now()
        self.end = datetime.now()
        self.first_lap = True
        self.lap = datetime.now()
        self.last_lap = datetime.now()
    
    def start_timer(self):
        self.start = datetime.now()
        start_time = self.start.strftime("%d/%m/%Y %H:%M:%S")
        self.last_lap = self.start
        return start_time

    def lap_timer(self):
        if not self.first_lap:
            self.last_lap = self.lap
        else:
            self.first_lap = False
        self.lap = datetime.now()
        lap_time = self.lap.strftime("%d/%m/%Y %H:%M:%S")
        duration = self.lap - self.last_lap
        duration_in_s = duration.total_seconds()
        days = divmod(duration_in_s, 86400)
        hours = divmod(days[1], 3600)
        minutes = divmod(hours[1], 60)
        seconds = divmod(minutes[1], 1)
        return lap_time, days[0], hours[0], minutes[0], seconds[0]

    def get_total_duration(self):
        self.end = datetime.now()
        end_time = self.end.strftime("%d/%m/%Y %H:%M:%S")
        duration = self.end - self.start
        duration_in_s = duration.total_seconds()
        days = divmod(duration_in_s, 86400)
        hours = divmod(days[1], 3600)
        minutes = divmod(hours[1], 60)
        seconds = divmod(minutes[1], 1)
        return end_time, days[0], hours[0], minutes[0], seconds[0]
