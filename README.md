# 松尾研LLM開発コンペ2025 開発コード・手順の標準化


@T41K1
PPOからGRPOへ変更する

以下、Bridge LLM開発コンペ2025におけるLLM開発コード・手順の標準化です。  
**「サーバ利用手順」は、サーバ利用における重要な注意事項も含まれるため必ずご確認ください。**  
**「LLM評価手順」は、開発したLLMのランキングに使うため必ずご確認ください。**

## サーバ利用手順

サーバにログインしてジョブシステムを利用する手順です。  
[こちら](https://docs.google.com/document/d/16KKkFM8Sbqx0wgcCY4kBKR6Kik01T-jn892e_y67vbM/edit?tab=t.0)

## LLM学習手順

LLM学習手順です。ファインチューニング，強化学習を含みます。  
[こちら](train/README.md)  

＊Huggingfaceから既存のデータセットをダウンロードして学習に利用する前提のコードになっています。独自にデータ収集・加工を行う場合は別途コードをご準備ください。参考までに2024コンペ時のコードは[こちら](https://github.com/matsuolab/ucllm_nedo_prod/tree/main)ですが、事前学習用ですのでご注意ください。

## LLM評価手順

LLM評価手順です。  
[Humanity's Last Exam](eval_hle/README.md)  
[Do Not Answer](eval_dna/README.md)  
[運営でdan及びhleの動作確認済みパッケージリスト](package-list.txt)

## Contributors

```
```

