# Predicting Brazilian court decisions


Repository of the paper entitled _Predicting Brazilian court decisions_.

### Versions

- [2019-04-20 Technical Report @ arXiv](https://arxiv.org/abs/1905.10348)
- [2019-04-21 Technical Report @ Github](https://github.com/proflage/technical-reports/blob/020fe07c06fc551a1305055a07c806f930a39fae/2019-04-21-Predicting_Brazilian_court_decisions.pdf)
- [2021 Full paper (TBA)]()

### Citation

- André Lage-Freitas, Héctor Allende-Cid, Orivaldo Santana, and Lívia Oliveira-Lage. _Predicting Brazilian court decisions_. Technical Report. 2019.
- [Bibtex citation TBA]()



# Methodology

We used this Python program for Steps 4, 5, 6, 7, 8, and 9 of our methodology. The Web scraper, regarding the Steps 1, 2, and 3 of our methodology, are under JusPredict (https://www.juspredict.com.br) Intellectual Property.

# Data set

This data set has Ementa (summary) decisions from the Tribunal de Justiça de Alagoas (TJAL, the State Supreme Court of Alagoas (Brazil), and their metadata. The file format is CSV and its separator (delimiter) is "<=>".

# Acknowledgments

- We thank the anonymous reviewers who provided significant and constructive critiques of this manuscript. 
- André Lage-Freitas, Héctor Allende-Cid, and Orivaldo Santana are founders of the artificial intelligence startup JusPredict. 
- Héctor Allende-Cid was funded by the Comisión Nacional de Investigación Científica y Tecnológica (CONICYT) under Grant No REDI170059.
- There was no additional external funding received for this study.

 
# Reproducible Materials

### Execution

```bash
python predicting-brazilian-court-decisions.py 
```

### Requirements

- Python 3.7.7
- Source-code `predicting-brazilian-court-decisions.py`
- Data `dataset.zip`
- Cuda-based GPU (required only for Deep Learning models)
- Required packages: 
  - absl-py==0.8.1
alembic==1.3.1
ase==3.21.1
asn1crypto==1.3.0
astor==0.8.0
async-generator==1.10
attrs==19.3.0
autovizwidget==0.13.1
backcall==0.1.0
beautifulsoup4==4.8.2
bert-serving-client==1.10.0
bert-serving-server==1.10.0
biterm==0.1.5
bleach==3.1.0
blinker==1.4
blis==0.2.4
boto==2.49.0
boto3==1.11.0
botocore==1.14.0
bs4==0.0.1
bz2file==0.98
celluloid==0.2.0
certifi==2019.11.28
certipy==0.1.3
cffi==1.13.2
chardet==3.0.4
cloudpickle==1.2.2
cryptography==2.8
cycler==0.10.0
cymem==2.0.2
Cython==0.29.14
cytoolz==0.9.0.1
dask==2.9.1
decorator==4.4.1
defusedxml==0.6.0
dill==0.2.9
docutils==0.15.2
entrypoints==0.3
es-core-news-sm==2.1.0
et-xmlfile==1.0.1
flags==0.0.1.2
funcy==1.14
future==0.18.2
gast==0.2.2
gensim==3.8.0
google-pasta==0.1.8
googledrivedownloader==0.4
GPUtil==1.4.0
grpcio==1.16.1
h5py==2.10.0
hdijupyterutils==0.12.9
idna==2.8
imageio==2.6.1
imbalanced-learn==0.6.2
imblearn==0.0
importlib-metadata==1.3.0
ipykernel==5.1.3
ipyparallel==6.2.4
ipython==7.13.0
ipython-genutils==0.2.0
ipywidgets==7.5.1
isodate==0.6.0
isort==4.3.21
jdcal==1.4.1
jedi==0.15.2
Jinja2==2.10.3
jmespath==0.9.4
joblib==0.14.1
json5==0.8.5
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==5.3.4
jupyter-console==6.0.0
jupyter-core==4.6.1
jupyter-lsp==0.8.0
jupyter-telemetry==0.0.5
jupyterhub==1.1.0
jupyterhub-systemdspawner==0.13
jupyterlab==2.1.2
jupyterlab-server==1.1.1
Keras==2.2.4
Keras-Applications==1.0.8
keras-contrib==2.0.8
Keras-Preprocessing==1.1.0
kiwisolver==1.1.0
langdetect==1.0.7
lazy-object-proxy==1.4.3
llvmlite==0.36.0
Mako==1.1.0
Markdown==3.1.1
MarkupSafe==1.1.1
mat4py==0.4.2
matplotlib==3.1.2
mccabe==0.6.1
mistune==0.8.4
mkl-fft==1.0.6
mkl-random==1.0.1
mock==3.0.5
more-itertools==8.0.2
mrcnn==0.2
msgpack==0.6.1
msgpack-numpy==0.4.3.2
murmurhash==1.0.2
nbconvert==5.6.1
nbformat==4.4.0
networkx==2.4
nltk==3.4.5
notebook==6.0.3
numba==0.53.1
numexpr==2.7.1
numpy==1.15.4
oauthlib==3.1.0
olefile==0.46
openpyxl==3.0.6
opt-einsum==3.1.0
packaging==20.0
pamela==1.0.0
pandas==0.25.3
pandocfilters==1.4.2
parso==0.5.2
patsy==0.5.1
pexpect==4.7.0
pickleshare==0.7.5
Pillow==7.0.0
plac==0.9.6
plotly==4.6.0
pluggy==0.13.1
preshed==2.0.1
prometheus-client==0.7.1
prompt-toolkit==2.0.10
protobuf==3.11.2
psycopg2==2.8.5
ptvsd==4.3.2
ptyprocess==0.6.0
py==1.8.1
pycparser==2.19
pycurl==7.43.0.3
pydocstyle==5.0.2
pydot==1.3.0
pydotplus==2.0.2
pyflakes==2.1.1
pygifsicle==1.0.1
Pygments==2.5.2
PyJWT==1.7.1
pykerberos==1.2.1
pyLDAvis==2.1.2
pymongo==3.9.0
pyOpenSSL==19.1.0
pyparsing==2.4.6
PyQt5==5.12.3
PyQt5-sip==4.19.18
PyQtWebEngine==5.12.1
pyrsistent==0.15.6
PySocks==1.7.1
pytest==5.3.4
python-dateutil==2.8.1
python-editor==1.0.4
python-json-logger==0.1.11
python-jsonrpc-server==0.3.4
python-louvain==0.15
pytz==2019.3
PyWavelets==1.1.1
PyYAML==5.2
pyzmq==18.1.0
qtconsole==4.6.0
rdflib==5.0.0
regex==2019.12.9
requests==2.22.0
requests-kerberos==0.12.0
retrying==1.3.3
ruamel.yaml==0.16.6
ruamel.yaml.clib==0.2.0
s3transfer==0.3.0
scikit-fuzzy==0.4.2
scikit-image==0.15.0
scikit-learn==0.22.2.post1
scipy==1.1.0
seaborn==0.9.0
Send2Trash==1.5.0
sentencepiece==0.1.85
six==1.13.0
smart-open==1.9.0
snowballstemmer==2.0.0
soupsieve==2.0
spacy==2.1.8
sparkmagic==0.13.1
SQLAlchemy==1.3.12
srsly==1.0.2
statsmodels==0.11.0
TBB==0.1
tensorboard==1.14.0
tensorflow==1.14.0
tensorflow-addons==0.8.3
tensorflow-estimator==1.14.0
tensorflow-hub==0.9.0
termcolor==1.1.0
terminado==0.8.3
testpath==0.4.4
thinc==7.0.8
toolz==0.10.0
torch-geometric==1.7.0
tornado==6.0.3
tqdm==4.41.1
traitlets==4.3.3
ujson==1.35
Unidecode==1.1.1
urllib3==1.25.7
wasabi==0.8.0
wcwidth==0.1.7
webencodings==0.5.1
Werkzeug==0.16.0
widgetsnbextension==3.5.1
wordcloud==1.6.0.post14+g1fc6868
wrapt==1.12.1
xgboost==0.90
xlrd==1.2.0
zipp==0.6.0
