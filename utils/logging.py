from collections import defaultdict
import logging
import numpy as np

class WandbHandler(logging.Handler):
    """Custom logging handler that sends console messages to wandb"""
    def __init__(self, wandb_module):
        super().__init__()
        self.wandb = wandb_module

    def emit(self, record):
        try:
            msg = self.format(record)
            # Log console messages to wandb as text
            if self.wandb.run is not None:
                self.wandb.log({"console_log": msg})
        except Exception:
            self.handleError(record)

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.use_wandb = False
        self.wandb = None

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_wandb(self, project_name=None, config=None, **kwargs):
        """Setup wandb logger and add handler to console logger"""
        try:
            import wandb
            self.wandb = wandb

            # Initialize wandb if not already initialized
            if wandb.run is None:
                wandb.init(project=project_name, config=config, **kwargs)

            self.use_wandb = True

            # Add wandb handler to console logger to capture all console messages
            wandb_handler = WandbHandler(wandb)
            # Use the same formatter as console logger if available
            if self.console_logger.handlers:
                formatter = self.console_logger.handlers[0].formatter
                if formatter:
                    wandb_handler.setFormatter(formatter)
            self.console_logger.addHandler(wandb_handler)

            self.console_logger.info("Wandb logging enabled. Project: {}".format(project_name or "default"))
        except ImportError:
            self.console_logger.warning("wandb not installed. Install it with: pip install wandb")
        except Exception as e:
            self.console_logger.warning("Failed to setup wandb: {}".format(e))

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

        # Log to wandb
        if self.use_wandb and self.wandb and self.wandb.run is not None:
            try:
                # Convert tensor to scalar if needed
                if hasattr(value, 'item'):
                    value = value.item()
                self.wandb.log({key: value}, step=t)
            except Exception as e:
                # Silently fail if wandb logging fails
                pass

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

