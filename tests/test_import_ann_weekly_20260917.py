"""Safety tests use tiny synthetic archives, not scientific observations."""
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import tempfile
import unittest
from zipfile import ZipFile, ZipInfo

SOURCE = Path(__file__).resolve().parents[1] / 'research/ftir_hips_chem/workflows/import_ann_weekly_20260917.py'
spec = importlib.util.spec_from_file_location('weekly_import', SOURCE)
weekly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weekly)


class ImportTests(unittest.TestCase):
    def fixture(self, root):
        pptx = io.BytesIO()
        with ZipFile(pptx, 'w') as z:
            for i in range(1, 27):
                z.writestr(f'ppt/slides/slide{i}.xml', '<slide/>')
                z.writestr(f'ppt/notesSlides/notesSlide{i}.xml', '<notes/>')
        data = {'deck.pptx': pptx.getvalue(), 'deck.pdf': b'%PDF-1.7\nfixture', 'Presenter_Notes.md': b'notes\n'}
        archive = root / 'package.zip'
        with ZipFile(archive, 'w') as z:
            for name, value in data.items():
                z.writestr('approved/' + name, value)
        manifest = {
            'source_archive_size_bytes':archive.stat().st_size,
            'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_root':'approved', 'main_slides':18, 'backup_slides':8,
            'artifacts':{name:{'size_bytes':len(value), 'sha256':hashlib.sha256(value).hexdigest()} for name, value in data.items()}
        }
        return archive, manifest

    def test_check_writes_nothing_and_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest = self.fixture(root)
            self.assertEqual(weekly.install(archive, root, manifest, True)['files_to_copy'], 3)
            self.assertFalse((root / weekly.DELIVERABLE).exists())
            self.assertEqual(weekly.install(archive, root, manifest)['files_copied'], 3)
            self.assertEqual(weekly.install(archive, root, manifest)['files_copied'], 0)

    def test_bad_archive_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest = self.fixture(root)
            manifest['source_archive_sha256'] = '0' * 64
            with self.assertRaises(ValueError): weekly.install(archive, root, manifest)
            self.assertFalse((root / weekly.DELIVERABLE).exists())

    def test_conflict_aborts_before_other_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest = self.fixture(root)
            target = root / weekly.DELIVERABLE / 'bundle'
            target.mkdir(parents=True); (target / 'Presenter_Notes.md').write_text('changed')
            with self.assertRaises(FileExistsError): weekly.install(archive, root, manifest)
            self.assertFalse((target / 'deck.pptx').exists())

    def test_member_paths(self):
        for name in ['approved/../x', '/approved/x', 'other/x', 'approved/C:x', 'approved/a\\b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                weekly.validate_member(ZipInfo(name), 'approved')

    def test_zip_symlink(self):
        info = ZipInfo('approved/link'); info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError): weekly.validate_member(info, 'approved')

    def test_destination_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest = self.fixture(root)
            real = root / 'real'; real.mkdir()
            target = root / weekly.DELIVERABLE; target.mkdir(parents=True)
            (target / 'bundle').symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError): weekly.install(archive, root, manifest)
            self.assertEqual(list(real.iterdir()), [])

    def test_wrong_primary_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest = self.fixture(root)
            manifest['artifacts']['deck.pptx']['sha256'] = '0' * 64
            with self.assertRaises(ValueError): weekly.install(archive, root, manifest)

    def test_incorrect_slide_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest = self.fixture(root)
            manifest['backup_slides'] = 7
            with self.assertRaises(ValueError): weekly.install(archive, root, manifest)


if __name__ == '__main__':
    unittest.main()
