import webuiapi


def connect(host='3088c0a2998cd045c7.gradio.live',
            port=7860,
            sampler='Euler a',
            steps=20):
    ser = webuiapi.WebUIApi(host=host, port=port, sampler=sampler, steps=steps, use_https=True)
    ser.set_auth('test', '123456')
    return ser


if __name__ == '__main__':
    conn = connect()
    ret = conn.txt2img(prompt="cute squirrel",
                    negative_prompt="ugly, out of frame",
                    seed=1003,
                    styles=["anime"],
                    cfg_scale=7)
    ret.image