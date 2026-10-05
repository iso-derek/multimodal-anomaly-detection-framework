from pathlib import Path
import sys,json,io,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import streamlit as st
from src.robustness_research import synthetic_cases,run_study
st.set_page_config(page_title='Fusion robustness lab',layout='wide')
st.title('Multimodal fusion robustness lab')
st.caption('One row per aligned event. Calibration learns modality weights; validation fixes thresholds; test cases measure robustness.')
source=st.sidebar.file_uploader('Aligned score CSV',type='csv')
st.info('The default benchmark is synthetic. It tests fusion mechanics, not trained detector accuracy or clinical/surveillance effectiveness. Synthetic quality values know the injected corruption.')
@st.cache_data
def evaluate(frame):return run_study(frame)
frame=pd.read_csv(source) if source else synthetic_cases()
try:metrics,predictions,metadata=evaluate(frame)
except ValueError as exc:st.error(str(exc));st.stop()
scenario=st.selectbox('Stress scenario',metrics.scenario.unique().tolist())
st.dataframe(metrics[metrics.scenario.eq(scenario)],hide_index=True)
st.line_chart(metrics.pivot(index='scenario',columns='method',values='auc'))
st.caption('Missing modalities are excluded. No reliable evidence produces abstention. Compare quality-unknown corruption to assess dependence on usable quality measurements.')
with st.expander('Frozen settings and data contract'):
    st.json(metadata)
    st.write('Required: case_id, group_id, split (calibration/validation/test), label (0/1), score_ and quality_ columns for tabular, timeseries, image and video. Detector scores must themselves be out of sample. Groups may not cross splits. Scores may be missing; quality must be in [0,1].')
    st.dataframe(frame.head())
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for name,data in [('cases.csv',frame),('metrics.csv',metrics),('predictions.csv',predictions)]:archive.writestr(name,data.to_csv(index=False))
    archive.writestr('metadata.json',json.dumps(metadata,indent=2))
st.download_button('Download reproducible study',buffer.getvalue(),'fusion_study.zip','application/zip')
