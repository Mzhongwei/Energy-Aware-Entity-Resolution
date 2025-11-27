from prometheus_client import Counter, Gauge

class Metrics:
    def __init__(self):
        self.messages_consumed = Counter("kafka_messages_consumed", "Total messages consumed")
        self.window_data_processing_time = Gauge("kafka_window_data_processing_time", "Time taken to process each message in the window")
        # self.data_buffer_time = Gauge("kafka_data_buffer_time", "Time taken to stay in buffer for each message")
        self.lag_metric = Gauge("kafka_consumer_lag", "Kafka consumer lag")
        
    
    def update_message_consumed(self):
        self.messages_consumed.inc()

    def update_window_data_processing_time(self, time):
        self.window_data_processing_time.set(time)
    
    # def update_data_buffer_time(self, time):
    #     self.data_buffer_time.set(time)

    def update_lag_metrics(self, lag):
        self.lag_metric.set(lag)
    