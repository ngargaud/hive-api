import requests
import json
import time
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
        if name in ["asr", "tts", "voicereco", "facereco"]:
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
        apis = ["asr", "tts", "voicereco", "facereco"]
        assert name in apis, "name must be in {}".format(apis)
        client = self.get_client(name)
        if client:
            return client.predict(api_name="/get_settings")


    def get_asr_language(self):
        client = self.get_client("asr")
        if client:
            return client.predict(api_name="/get_language")


    def set_asr_task(self, task="transcribe"):
        tasks = ["transcribe", "translate"]
        assert task in tasks, "task must be in {}".format(tasks)
        value = task == "translate"
        client = self.get_client("asr")
        if client:
            client.predict(value=value, api_name="/set_asr_task")


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
                output = data.get("value")
                return self.fetch_tts_file(output)
            else:
                return client.submit(message=text, api_name="/chat_request")
        return None


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


    def call_asr(self, filename, wait=True):
        client = self.get_client("asr")
        if client:
            if wait:
                return client.predict(filename=gc.handle_file(filename), task="transcribe", api_name="/process")
            else:
                # runs the prediction in a background thread
                return client.submit(filename=gc.handle_file(filename), task="transcribe", api_name="/process")


    def call_voice_compare(self, embed, embed_ref, wait=True):
        client = self.get_client("voicereco")
        if client:
            if wait:
                return client.predict(embedding=embed, embedding_ref=embed_ref, api_name="/compare_embeddings")
            else:
                # runs the prediction in a background thread
                return client.submit(embedding=embed, embedding_ref=embed_ref, api_name="/compare_embeddings")


    def call_voice_reco(self, filename, wait=True):
        client = self.get_client("voicereco")
        if client:
            if wait:
                return client.predict(filename=gc.handle_file(filename), api_name="/process")
            else:
                # runs the prediction in a background thread
                return client.submit(filename=gc.handle_file(filename), api_name="/process")


    def call_face_reco(self, filename, wait=True):
        client = self.get_client("facereco")
        if client:
            if wait:
                return client.predict(filename=gc.handle_file(filename), api_name="/process")
            else:
                # runs the prediction in a background thread
                return client.submit(filename=gc.handle_file(filename), api_name="/process")


    def create_metadata(self, audio=None, image=None):
        start = time.time()
        results = {}
        if audio:
            audio_process = self.call_asr(audio, wait=False)
            voice_process = self.call_voice_reco(audio, wait=False)
        if image:
            face_process =  self.call_face_reco(image, wait=False)

        if audio:
            if audio_process:
                try:
                    results["asr"] = json.loads(audio_process.result())
                except:
                    results["asr"] = audio_process.result()
            if voice_process:
                try:
                    results["voices"] = json.loads(voice_process.result())
                except:
                    results["voices"] = voice_process.result()
        if image and face_process:
            try:
                results["faces"] = json.loads(face_process.result())
            except:
                results["faces"] = face_process.result()
        print("[main]   metadata generation:", time.time() - start, "seconds")
        return results
