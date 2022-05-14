import os


class Config(object):
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    ENV = "development"
    DATABASE_URI = os.environ['DATABASE_URL']
    DEBUG = True
    MAPBOX_KEY = '123'