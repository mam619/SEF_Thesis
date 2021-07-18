from datetime import datetime
import pytz

TUNING_DATES = {
    "start": datetime(2018, 4, 1, tzinfo=pytz.utc),
    "end": datetime(2018, 11, 15, tzinfo=pytz.utc),
}

TEST_DATES = {
    "start": datetime(2018, 5, 15, tzinfo=pytz.utc),
    "end": datetime(2019, 1, 1, tzinfo=pytz.utc),
}
