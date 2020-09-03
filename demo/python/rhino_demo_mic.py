import argparse
import os
import struct
import sys
from threading import Thread
import wave
import numpy as np
import pyaudio
import soundfile
import audioop
from moviepy.editor import *
import librosa
import csv


def Video_to_Audio(path):
    if ".wav" in path:
        return path
    else:

        VideoPath = path
        AudioPath = path.replace("mp4", "wav")
        video = VideoFileClip(VideoPath)
        audio = video.audio
        audio.write_audiofile(AudioPath)
        print(AudioPath)

        data, rate = librosa.load(AudioPath, sr=16000)
        data, samplerate = data, rate
        soundfile.write(AudioPath, data, samplerate)
        return AudioPath


class RhinoDemo(Thread):
    def __init__(
        self,
        rhino_library_path,
        rhino_model_file_path,
        rhino_context_file_path,
        porcupine_library_path,
        porcupine_model_file_path,
        porcupine_keyword_file_path,
        input_device_index=None,
        output_path=None,
        video_path=None,
    ):
        """
        Constructor.

        :param rhino_library_path: Absolute path to Rhino's dynamic library.
        :param rhino_model_file_path: Absolute path to Rhino's model parameter file.
        :param rhino_context_file_path: Absolute path to Rhino's context file that defines the context of commands.
        :param porcupine_library_path: Absolute path to Porcupine's dynamic library.
        :param porcupine_model_file_path: Absolute path to Porcupine's model parameter file.
        :param porcupine_keyword_file_path: Absolute path to Porcupine's keyword file for wake phrase.
        :param input_device_index: Optional argument. If provided, audio is recorded from this input device. Otherwise,
        the default audio input device is used.
        :param output_path: If provided recorded audio will be stored in this location at the end of the run.
        """

        super(RhinoDemo, self).__init__()

        self._rhino_library_path = rhino_library_path
        self._rhino_model_file_path = rhino_model_file_path
        self._rhino_context_file_path = rhino_context_file_path

        self._porcupine_library_path = porcupine_library_path
        self._porcupine_model_file_path = porcupine_model_file_path
        self._porcupine_keyword_file_path = porcupine_keyword_file_path

        self._input_device_index = input_device_index

        self._output_path = output_path
        self._video_path = video_path
        if self._output_path is not None:
            self._recorded_frames = list()

    def run(self):
        def _frame_index_to_sec(frame_index):
            return (
                float(frame_index * rhino.frame_length) / float(rhino.sample_rate)
            ) - float(1)

        """
         Creates an input audio stream, initializes wake word detection (Porcupine) and speech to intent (Rhino)
         engines, and monitors the audio stream for occurrences of the wake word and then infers the intent from speech
         command that follows.
         """

        porcupine = None
        rhino = None
        pa = None
        audio_stream = None

        wake_phrase_detected = True
        intent_extraction_is_finalized = False
        Apath = Video_to_Audio(self._video_path)
        wf = wave.Wave_read(Apath)
        ww, sr = soundfile.read(Video_to_Audio(self._video_path))
        print(len(ww))
        try:
            porcupine = Porcupine(
                library_path=self._porcupine_library_path,
                model_file_path=self._porcupine_model_file_path,
                keyword_file_paths=[self._porcupine_keyword_file_path],
                sensitivities=[0.5],
            )

            rhino = Rhino(
                library_path=self._rhino_library_path,
                model_path=self._rhino_model_file_path,
                context_path=self._rhino_context_file_path,
                sensitivity=0.6,
            )

            print()
            print(
                "****************************** context ******************************"
            )
            print(rhino.context_info)
            print(
                "*********************************************************************"
            )
            print()

            pa = pyaudio.PyAudio()

            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
                input_device_index=self._input_device_index,
            )

            test = 0
            Tpath = Apath.replace("wav", "txt")
            f = open(Tpath, "w")

            ouput = ""
            classtr = ""
            startcount = 0
            endcount = 0
            cango = 1
            checkfirst = 0
            data_csv = [["Class_num", "Start_time", "End_time"]]
            ClassNum = None
            Start_time = None
            Start_time2 = None
            ClassNum2 = None
            rm = None
            # NOTE: This is true now and will be correct possibly forever. If it changes the logic below need to change.
            assert porcupine.frame_length == rhino.frame_length
            try:
                while True:

                    date = wf.readframes(porcupine.frame_length)
                    pcm = audio_stream.read(
                        porcupine.frame_length, exception_on_overflow=False
                    )

                    pcm = struct.unpack_from("h" * porcupine.frame_length, date)

                    if self._output_path is not None:
                        self._recorded_frames.append(pcm)

                    if not wake_phrase_detected:
                        wake_phrase_detected = porcupine.process(pcm)

                        if wake_phrase_detected:
                            print("detected wake phrase")
                    elif not intent_extraction_is_finalized:
                        intent_extraction_is_finalized = rhino.process(pcm)

                    else:

                        if rhino.is_understood():
                            cango = 1
                            intent, slot_values = rhino.get_intent()
                            print()
                            if intent == "EndWork":

                                endcount += 1
                                classstr = " - %s" % _frame_index_to_sec(test)

                            else:
                                checkfirst += 1
                                startcount += 1
                                endcount = 0
                                for slot, value in slot_values.items():
                                    print("%s: %s" % (slot, value))
                                    classstr = ("%s: %s" % (slot, value)) + (
                                        " start time is %s" % _frame_index_to_sec(test)
                                    )
                                    if startcount == 2:
                                        Start_time2 = Start_time
                                        ClassNum2 = ClassNum
                                    Start_time = _frame_index_to_sec(test)
                                    ClassNum = value
                            print()

                            print(
                                "intent : %s at time: %f"
                                % (intent, _frame_index_to_sec(test))
                            )
                            print()
                        else:
                            print("didn't understand the command")
                            cango = 0

                        rhino.reset()
                        wake_phrase_detected = True
                        intent_extraction_is_finalized = False
                        print(startcount, endcount)
                        print(ouput)

                        if cango:
                            if endcount == 1 and startcount == 0:
                                ouput = classstr
                                f.write("-1 class end at" + ouput + "\n")
                                endcount = 0
                                ouput = ""
                                data_csv.append(["-1", "-1", _frame_index_to_sec(test)])
                            elif ouput == "" and endcount == 0 and startcount == 1:
                                ouput = classstr

                            elif ouput != "" and endcount == 1:
                                try:
                                    data_csv.remove(rm)
                                except:
                                    pass
                                data_csv.append(
                                    [ClassNum, Start_time, _frame_index_to_sec(test)]
                                )
                                ouput += classstr
                                endcount = 0
                                startcount = 0
                                f.write(ouput + "\n")
                                ouput = ""
                            elif endcount == 0 and startcount == 2:
                                if checkfirst == 2:
                                    data_csv.append([ClassNum2, Start_time2, "-1"])

                                    f.write(ouput + "\n")
                                data_csv.append([ClassNum, Start_time, "-1"])
                                rm = [ClassNum, Start_time, "-1"]
                                ouput = classstr
                                f.write(ouput + "\n")
                                startcount = 1

                    test += 1
            except:
                print("EOF")
                print(_frame_index_to_sec(test))
                data_csv.append(["Maybe miss", classstr, classstr])
                f.write("Могла быть упущенная метка : %s" % classstr)
                with open("sw_data_new.csv", "w") as f:
                    writer = csv.writer(f)
                    for row in data_csv:
                        writer.writerow(row)

        except KeyboardInterrupt:
            print("stopping ...")

        finally:
            if porcupine is not None:
                porcupine.delete()

            if rhino is not None:
                rhino.delete()

            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

            if self._output_path is not None and len(self._recorded_frames) > 0:
                recorded_audio = np.concatenate(self._recorded_frames, axis=0).astype(
                    np.int16
                )
                soundfile.write(
                    os.path.expanduser(self._output_path),
                    recorded_audio,
                    samplerate=porcupine.sample_rate,
                    subtype="PCM_16",
                )

    _AUDIO_DEVICE_INFO_KEYS = ["index", "name", "defaultSampleRate", "maxInputChannels"]

    @classmethod
    def show_audio_devices_info(cls):
        """ Provides information regarding different audio devices available. """

        pa = pyaudio.PyAudio()

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(
                ", ".join(
                    "'%s': '%s'" % (k, str(info[k]))
                    for k in cls._AUDIO_DEVICE_INFO_KEYS
                )
            )

        pa.terminate()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rhino_library_path",
        default=RHINO_LIBRARY_PATH,
        help="absolute path to Rhino's dynamic library",
    )

    parser.add_argument(
        "--rhino_model_file_path",
        default=RHINO_MODEL_FILE_PATH,
        help="absolute path to Rhino's model file path",
    )

    parser.add_argument(
        "--rhino_context_file_path", help="absolute path to Rhino's context file"
    )

    parser.add_argument(
        "--porcupine_library_path",
        default=PORCUPINE_LIBRARY_PATH,
        help="absolute path to Porcupine's dynamic library",
    )

    parser.add_argument(
        "--porcupine_model_file_path",
        default=PORCUPINE_MODEL_FILE_PATH,
        help="absolute path to Porcupine's model parameter file",
    )

    parser.add_argument(
        "--porcupine_keyword_file_path",
        default=KEYWORD_FILE_PATHS["picovoice"],
        help="absolute path to porcupine keyword file",
    )

    parser.add_argument(
        "--input_audio_device_index",
        help="index of input audio device",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_path",
        help="absolute path to where recorded audio will be stored. If not set, it will be bypassed.",
        default=None,
    )
    parser.add_argument("--video_path", help="Path_to_video.", default=None)

    parser.add_argument("--show_audio_devices_info", action="store_true")

    args = parser.parse_args()

    if args.show_audio_devices_info:
        RhinoDemo.show_audio_devices_info()
    else:
        if not args.rhino_context_file_path:
            raise ValueError("missing rhino_context_file_path")

        RhinoDemo(
            rhino_library_path=args.rhino_library_path,
            rhino_model_file_path=args.rhino_model_file_path,
            rhino_context_file_path=args.rhino_context_file_path,
            porcupine_library_path=args.porcupine_library_path,
            porcupine_model_file_path=args.porcupine_model_file_path,
            porcupine_keyword_file_path=args.porcupine_keyword_file_path,
            input_device_index=args.input_audio_device_index,
            output_path=args.output_path,
            video_path=args.video_path,
        ).run()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../binding/python"))
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__), "../../resources/porcupine/binding/python"
        )
    )
    sys.path.append(
        os.path.join(os.path.dirname(__file__), "../../resources/util/python")
    )
    from porcupine import Porcupine
    from rhino import Rhino
    from util import *

    main()
