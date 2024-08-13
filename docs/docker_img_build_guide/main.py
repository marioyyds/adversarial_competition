import sys

print(f"接收到的命令参数：{sys.argv}")

from absl import flags

flags.DEFINE_string(name="origin_sample_path", default=None, help="原始样本集路径")
flags.DEFINE_string(name="attack_sample_path", default=None, help="攻击样本集路径")
flags.DEFINE_integer(name="sample_type", default=1, help="样本类型：1-可见光、2-遥感")
FLAGS = flags.FLAGS
FLAGS(sys.argv)

print(f"原始样本集路径：{FLAGS.origin_sample_path}")
print(f"攻击样本集路径：{FLAGS.attack_sample_path}")
print(f"样本类型：{FLAGS.sample_type}")
