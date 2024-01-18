import datadog
from datadog import initialize

# initialize()
initialize(statsd_host="172.17.0.1", statsd_port=8125)


class DataDogUtils():

    @staticmethod
    def increment(stat, tags=[], value=1):
        datadog.statsd.increment(stat, value=value, tags=tags)

    @staticmethod
    def gauge(stat, value, tags=[]):
        datadog.statsd.gauge(stat, value, tags)

    @staticmethod
    def histogram(stat, value, tags=[]):
        datadog.statsd.histogram(stat, value, tags)