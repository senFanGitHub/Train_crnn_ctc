#!/usr/bin/env python
# -*- coding: utf-8 -*-
import mxnet as mx
from mxnet import nd,gluon
from mxnet.gluon import data
import json 
import time
import numpy as np
import cv2
import os
import random

import base64
import re
from PIL import Image

# Flask imports:
from flask import Flask, request, request_finished, json, abort, make_response, Response, jsonify

import sys
sys.path.append("./")
# logging
import logging
from logging.handlers import RotatingFileHandler


KEY="key"
MODEL_DIR="model_crnn"
CTX=mx.cpu()
    
from common import crnn_infer
net =crnn_infer(MODEL_DIR,ctx=CTX)



# The main application:
app = Flask(__name__)

IMAGE_DECODE_ERROR = 10
PREDICTION_ERROR = 12
INVALID_FORMAT = 30
INVALID_API_KEY = 31
MISSING_ARGUMENTS = 40

errors = {
    IMAGE_DECODE_ERROR: "IMAGE_DECODE_ERROR",
    PREDICTION_ERROR: "PREDICTION_ERROR",
    INVALID_FORMAT: "INVALID_FORMAT",
    INVALID_API_KEY: "INVALID_API_KEY",
    MISSING_ARGUMENTS: "MISSING_ARGUMENTS"
}

# Setup the logging for the server, so we can log all exceptions
# away. We also want to acquire a logger for the rec framework,
# so we can be sure, that all logging goes into one place.
LOG_FILENAME = 'server_online.log'
LOG_BACKUP_COUNT = 5
LOG_FILE_SIZE_BYTES = 50 * 1024 * 1024


def init_logger(app):
    handler = RotatingFileHandler(LOG_FILENAME, maxBytes=LOG_FILE_SIZE_BYTES, backupCount=LOG_BACKUP_COUNT)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    loggers = [app.logger, logging.getLogger('rec')]
    for logger in loggers:
        logger.addHandler(handler)


# Bring the model variable into global scope. This might be
# dangerous in Flask, I am trying to figure out, which is the
# best practice here.

# Initializes the Flask application, which is going to
# add the loggers, load the initial rec model and
# all of this.
def init_app(app):
    init_logger(app)


init_app(app)


@app.before_request
def log_request():
    app.logger.debug("Request: %s %s", request.method, request.url)
#     sys.stderr.write( request.url+'\n')


# The WebAppException might be useful. It enables us to
# throw exceptions at any place in the application and give the user
# a custom error code.
class WebAppException(Exception):
    def __init__(self, error_code, exception=None, status_code=None):
        Exception.__init__(self)
        self.status_code = 400
        self.exception = exception
        self.error_code = error_code
        try:
            self.message = errors[self.error_code]
        except:
            self.error_code = UNKNOWN_ERROR
            self.message = errors[self.error_code]
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self):
        rv = dict()
        rv['status'] = 'failed'
        rv['code'] = self.error_code
        rv['message'] = self.message
        return rv
  


# in a method and raise a new WebAppException with the
# original Exception included. This is a quick and dirty way
# to minimize error handling code in our server.
class ThrowsWebAppException(object):
    def __init__(self, error_code, status_code=None):
        self.error_code = error_code
        self.status_code = status_code

    def __call__(self, function):
        def returnfunction(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                raise WebAppException(self.error_code, e)

        return returnfunction


# Register an error handler on the WebAppException, so we
# can return the error as JSON back to the User. At the same
# time you should do some logging, so it doesn't pass by
# silently.
@app.errorhandler(WebAppException)
def handle_exception(error):
    app.logger.exception(error.exception)
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def read_image(base64_image):
    enc_data = base64.b64decode(base64_image)
    nparr = np.fromstring(enc_data, np.uint8)
    img_np = cv2.imdecode(nparr, 1)
    return img_np


def get_prediction(img_data, image_key):
    im = read_image(img_data)
    if im is None:
        raise WebAppException(error_code=IMAGE_DECODE_ERROR)

    try:
        if image_key == KEY:
            prediction = net.predict(im)   
            return prediction
        else:
            raise WebAppException(error_code=INVALID_API_KEY)
    except:
  
        raise WebAppException(error_code=PREDICTION_ERROR)



# Now add the API endpoints for recognizing, learning and
# so on. If you want to use this in any public setup, you
# should add rate limiting, auth tokens and so on.
@app.route('/api/recognize', methods=['GET', 'POST'])
def identify():

    if request.headers['Content-Type'] == 'application/json':
        try:
            image_data = request.json['image']
            if "key" in request.json:
                image_key = request.json['key']
            else: 
                raise WebAppException(error_code=INVALID_API_KEY)
        except:
            raise WebAppException(error_code=MISSING_ARGUMENTS)
        prediction = get_prediction(image_data, image_key)
        try:
            response = jsonify(res=prediction,
                               att=base64.b64decode(image_data) if prediction is 'INVALID_IMAGE_DATA_REQUEST' else '')
        except:
            response = jsonify(res=prediction, att='a non-jsonified file...')
            sys.stderr.write('Error: Request image data is a non-jsonified file=' + str(image_data) + '\n')
        return response
    else:
        raise WebAppException(error_code=INVALID_FORMAT)


if __name__ == "__main__":
    app.run()


    