import requests
import json
import os
import ollama as ol
import gradio_client as gc
import asyncio, asyncvnc
from PIL import Image

class HiveApi():
    def __init__(self, url=None):
        self.url = url
        self.internal_map = {
            "ollama": 'http://ollama:11434',
            "asr": 'http://ai_api_asr:8002',
            "tts": 'http://ai_api_tts:8002',
            "voicereco": 'http://ai_api_voicereco:8002',
            "facereco": 'http://ai_api_facereco:8002',
            "sandbox": "ai_sandbox:5901"
        }
        self.external_map = {
            "ollama": '{}:8060'.format(url),
            "asr": '{}:8063'.format(url),
            "tts": '{}:8064'.format(url),
            "voicereco": '{}:8065'.format(url),
            "facereco": '{}:8066'.format(url),
            "sandbox": '{}:8067'.format(url)
        }
        self.client = {
            "ollama": ol.Client(host=self.get_api_url("ollama"), verify=False),
            "asr": gc.Client(self.get_api_url("asr"), ssl_verify=False),
            "tts": gc.Client(self.get_api_url("tts"), ssl_verify=False),
            "voicereco": gc.Client(self.get_api_url("voicereco"), ssl_verify=False),
            "facereco": gc.Client(self.get_api_url("facereco"), ssl_verify=False)
        }

    def get_api_url(self, name):
        """
        Build the selected api url
        """
        if self.url:
            return self.external_map.get(name)
        else:
            return self.internal_map.get(name)


    def get_client(self, name):
        client = self.client.get(name)
        if client:
            return client
        else:
            raise Exception("Error {} client not found".format(name))


    def get_api_settings(self, name):
        # apis = ["asr", "tts"]
        apis = ["asr", "tts", "voicereco", "facereco"]
        assert name in apis, "name must be in {}".format(apis)
        return self.get_client(name).predict(api_name="/get_settings")


    def get_asr_language(self):
        return self.get_client("asr").predict(api_name="/get_language")


    def set_asr_task(self, task="transcribe"):
        tasks = ["transcribe", "translate"]
        assert task in tasks, "task must be in {}".format(tasks)
        value = task == "translate"
        self.get_client("asr").predict(value=value, api_name="/set_asr_task")


    def set_tts_lang(self, lang="en"):
        if lang in ["fr"]:
            lang = "fr-fr"
        langs = ["en", "fr-fr"]
        if not lang in langs:
            print("WARNING lang {} must be in {}".format(lang, langs))
        self.get_client("tts").predict(value=lang, api_name="/set_tts_language")


    def set_tts_clone(self, value=False):
        self.get_client("tts").predict(value=value, api_name="/set_tts_clone")


    def set_tts_clone_voice(self, filename):
        self.get_client("tts").predict(filename=gc.handle_file(filename), api_name="/set_file")


    def call_tts(self, text, wait=True):
        if wait:
            data = self.get_client("tts").predict(message=text, api_name="/chat_request")
            output = data.get("value")
            return self.fetch_tts_file(output)
        else:
            return self.get_client("tts").submit(message=text, api_name="/chat_request")


    def fetch_tts_file(self, remote_path):
        output_url = "{}/file={}".format(self.get_api_url("tts"), remote_path)
        head, tail = os.path.split(remote_path)
        if not os.path.exists(head):
            os.makedirs(head)
        with requests.get(output_url, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(remote_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk:
                    f.write(chunk)
        return remote_path


    def call_asr(self, audio_file, wait=True):
        if wait:
            return self.get_client("asr").predict(audio_in=gc.handle_file(audio_file), api_name="/audio_request")
        else:
            # runs the prediction in a background thread
            return self.get_client("asr").submit(audio_in=gc.handle_file(audio_file), api_name="/audio_request")


    async def get_sandbox_screenshot(self, filename=None, client=None):
        if client:
            pixels = await client.screenshot()
            image = Image.fromarray(pixels)
            if filename:
                image.save(filename)
            return image
        else:
            url, port = self.get_api_url("sandbox").rsplit(":", 1)
            async with asyncvnc.connect(url, int(port)) as client:
                pixels = await client.screenshot()
                image = Image.fromarray(pixels)
                if filename:
                    image.save(filename)
                return image


    def call_voice_reco(self, filename, wait=True):
        if wait:
            return self.get_client("voicereco").predict(filename=gc.handle_file(filename), api_name="/audio_request")
        else:
            # runs the prediction in a background thread
            return self.get_client("voicereco").submit(filename=gc.handle_file(filename), api_name="/audio_request")


    def call_voice_compare(self, embed, embed_ref, wait=True):
        if wait:
            return self.get_client("voicereco").predict(embedding=embed, embedding_ref=embed_ref, api_name="/compare_embeddings")
        else:
            # runs the prediction in a background thread
            return self.get_client("voicereco").submit(embedding=embed, embedding_ref=embed_ref, api_name="/compare_embeddings")



    def call_face_reco(self, filename, wait=True):
        if wait:
            return self.get_client("facereco").predict(filename=gc.handle_file(filename), api_name="/set_file")
        else:
            # runs the prediction in a background thread
            return self.get_client("facereco").submit(filename=gc.handle_file(filename), api_name="/set_file")
