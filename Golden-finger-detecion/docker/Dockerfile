FROM openvino/ubuntu18_dev:2021.4.2
# ------------------------------------------------------------------
WORKDIR /workspace
COPY ["./format_print.sh", "./requirements.sh", "/workspace/" ]
USER root
# ------------------------------------------------------------------
# 由於 安裝套件有互動界面會導致報錯 所以先行關閉
ENV DEBIAN_FRONTEND noninteractive  
RUN chmod +x ./requirements.sh \
&& ./requirements.sh \
&& rm ./*.sh
# ------------------------------------------------------------------
# 寫入進入W
ENTRYPOINT [ "/bin/bash", "-c" ]
