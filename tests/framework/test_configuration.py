from pathlib import Path
import tempfile
import unittest

from desim.framework.configuration import AppConfigLoader, ConfigurationError


class TestConfiguration(unittest.TestCase):
    def test_load_from_dict(self) -> None:
        loader = AppConfigLoader()
        cfg = loader.load_from_dict(
            {
                "dataset": {"source": "data/dataset.json", "format": "json"},
                "simulation": {"until": 20, "max_events": 1000},
                "scheduler": {"name": "random", "options": {"seed": 42}},
                "energy": {"alpha_share": 0.8},
                "metrics": {
                    "sla_lambda": 1,
                    "sla_theta": 1,
                    "sla_eta_max": 2,
                    "fairness_omega_1": 0.5,
                    "fairness_omega_2": 0.5,
                    "fairness_mu": 1,
                    "fitness_w_energy": 0.6,
                    "fitness_w_sla": 0.4,
                    "fitness_xi": 1,
                },
                "random_seeds": {"global_seed": 123, "scheduler_seed": 456},
                "random_benchmark": {"enabled": True, "sample_count": 50},
                "plugins": [
                    {
                        "name": "custom_mutation",
                        "enabled": True,
                        "entrypoint": "pkg.module:Class",
                        "options": {"rate": 0.1},
                    }
                ],
            }
        )

        self.assertEqual(cfg.dataset.source, "data/dataset.json")
        self.assertEqual(cfg.scheduler.name, "random")
        self.assertEqual(cfg.random_seeds.scheduler_seed, 456)
        self.assertEqual(cfg.random_benchmark.sample_count, 50)
        self.assertEqual(len(cfg.plugins), 1)
        self.assertEqual(cfg.plugins[0].name, "custom_mutation")

    def test_load_from_yaml(self) -> None:
        loader = AppConfigLoader()

        yaml_text = """
        dataset:
          source: data/dataset.json
          format: json
        simulation:
          until: 20
        scheduler:
          name: random
          options:
            seed: 7
        energy:
          alpha_share: 1.0
        metrics:
          fairness_omega_1: 0.7
          fairness_omega_2: 0.3
        random_seeds:
          global_seed: 11
        random_benchmark:
          enabled: true
          sample_count: 75
        plugins:
          - name: plugin_a
            enabled: true
            entrypoint: pkg.plugin:main
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            cfg = loader.load_from_yaml(path)

        self.assertEqual(cfg.dataset.format, "json")
        self.assertEqual(cfg.scheduler.options["seed"], 7)
        self.assertEqual(cfg.random_benchmark.sample_count, 75)
        self.assertEqual(cfg.plugins[0].entrypoint, "pkg.plugin:main")

    def test_dataset_is_required(self) -> None:
        loader = AppConfigLoader()
        with self.assertRaises(ConfigurationError):
            loader.load_from_dict({"simulation": {}})

    def test_invalid_fairness_weights(self) -> None:
        loader = AppConfigLoader()
        with self.assertRaises(ConfigurationError):
            loader.load_from_dict(
                {
                    "dataset": {"source": "x"},
                    "metrics": {"fairness_omega_1": 0.2, "fairness_omega_2": 0.2},
                }
            )


if __name__ == "__main__":
    unittest.main()

