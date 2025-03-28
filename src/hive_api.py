import requests
import json
import time
import os
import ollama as ol
import gradio_client as gc
from PIL import Image

class HiveApi():
    def __init__(self, url=None):
        self.url = url
        self.internal_map = {
            "ollama": 'http://ollama:11434',
            "reco": 'http://ai_api_reco:8002',
            "tts": 'http://ai_api_tts:8002'
        }
        self.external_map = {
            "ollama": '{}:8036'.format(url),
            "tts": '{}:8037'.format(url),
            "reco": '{}:8038'.format(url)
        }
        self.clients = {}


    def get_api_url(self, name):
        """
        Build the selected api url
        """
        if self.url:
            return self.external_map.get(name)
        else:
            return self.internal_map.get(name)


    def create_client(self, name):
        client = None
        if name in ["reco", "tts"]:
            client = gc.Client(self.get_api_url(name), ssl_verify=False)
        elif name in ["ollama"]:
            client = ol.Client(host=self.get_api_url(name), verify=False)
        if client and not name in self.clients:
            self.clients[name] = client
        return client


    def get_client(self, name):
        try:
            client = self.clients.get(name)
            if client:
                return client
            else:
                client = self.create_client(name)
                if client:
                    return client
                raise Exception("Error {} client not found".format(name))
        except Exception as e:
            print(e)


    def get_api_settings(self, name):
        # apis = ["asr", "tts"]
        apis = ["asr", "tts"]
        assert name in apis, "name must be in {}".format(apis)
        client = self.get_client(name)
        if client:
            return client.predict(api_name="/get_settings")


    def set_tts_lang(self, lang="en"):
        if lang in ["fr"]:
            lang = "fr-fr"
        langs = ["en", "fr-fr"]
        if not lang in langs:
            print("WARNING lang {} must be in {}".format(lang, langs))
        client = self.get_client("tts")
        if client:
            client.predict(value=lang, api_name="/set_tts_language")


    def set_tts_clone(self, value=False):
        client = self.get_client("tts")
        if client:
            client.predict(value=value, api_name="/set_tts_clone")


    def set_tts_clone_voice(self, filename):
        client = self.get_client("tts")
        if client:
            client.predict(filename=gc.handle_file(filename), api_name="/set_file")


    def call_tts(self, text, wait=True):
        client = self.get_client("tts")
        if client:
            if wait:
                data = client.predict(message=text, api_name="/chat_request")
                return data.get("value")
            else:
                return client.submit(message=text, api_name="/chat_request")
        return None


    def call_reco(self, filename, tasks=None, wait=True):
        client = self.get_client("reco")
        start = time.time()
        if client:
            if wait:
                return client.predict(filename=gc.handle_file(filename), tasks=tasks, api_name="/process")
            else:
                # runs the prediction in a background thread
                return client.submit(filename=gc.handle_file(filename), tasks=tasks, api_name="/process")
        print("[main]   metadata generation:", time.time() - start, "seconds")
        return results
