from typing import List
import unittest

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Mixin(object):
    @declared_attr
    def __tablename__(cls):  # table name is lowercase class name
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True, autoincrement=True)

    def __len__(self):
        return len(self.columns)

    def __getitem__(self, index):
        return getattr(self, self.columns[index])

    def __setitem__(self, index, value):
        setattr(self, self.columns[index], value)

    def __repr__(self):
        s = 'id={}: '.format(self.id)
        s += ' '.join(['{}={}'.format(self.columns[i], self[i]) for i in range(len(self))])
        return s


class Dataset(Base, Mixin):
    name = Column(String, unique=True, nullable=False)
    # parameter = Column(Float, nullable=False)
    file = relationship('File', back_populates='dataset', cascade='all, delete, delete-orphan')  # cascade configuration
    columns = ['name']


class File(Base, Mixin):
    path = Column(String, unique=False, nullable=False)  # a file can belong to more than one dataset
    dataset_id = Column(Integer, ForeignKey('dataset.id'))
    # many-to-one
    dataset = relationship('Dataset', back_populates='file')
    columns = ['path']


class Model(object):
    def __init__(self, engine):
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)
        self.session = session()
        # discard any stale state from an unclean shutdown
        self.session.rollback()

    def add_dataset(self, dataset: Dataset):
        self.session.add(dataset)
        self.session.commit()

    def add_files(self, dataset_id, files: List[File]):
        d = self.session.query(Dataset).get(int(dataset_id))
        if d is not None:
            d.file.extend(files)
            self.session.add(d)
            self.session.commit()
        else:
            raise KeyError('dataset_id=%i does not exist' % dataset_id)

    def del_dataset(self, dataset_id):
        # TODO: int()?
        d = self.session.query(Dataset).get(int(dataset_id))
        if d is not None:
            self.session.delete(d)  # cascading delete
            self.session.commit()
        else:
            raise KeyError(f'dataset_id={dataset_id} does not exist')

    def del_file(self, file_id):
        # TODO: int()?
        f = self.session.query(File).get(int(file_id))
        if f is not None:
            self.session.delete(f)
            self.session.commit()
        else:
            raise KeyError('file_id=%i does not exist' % file_id)

    def get_dataset(self, dataset_id=None) -> Dataset:
        """
        returns links to the database objects, 
        allowing read, and write access for updating 
        - this needs model.session.commit() afterwards
        """
        q = self.session.query(Dataset)
        if dataset_id is None:
            return q.all()
        else:
            return q.filter(Dataset.id == dataset_id).first()

    def get_file(self, file_id=None, dataset_id=None) -> File:
        q = self.session.query(File)
        if dataset_id is None:
            if file_id is None:
                return q.all()
            else:
                return q.filter(File.id == file_id).first()
        else:
            assert file_id is None  # disregarding file_id value
            return q.filter(File.dataset_id == dataset_id).all()


class TestModel(unittest.TestCase):
    def setUp(self):
        from sqlalchemy import create_engine
        engine = create_engine('sqlite://',
                               echo=False)  # memory-based db
        self.model = Model(engine)

    def test_morebase(self):
        d = Dataset(name='one', parameter=0)
        f = File(path='file_path')
        s = str(d)
        s = str(f)

    def test_allthethings(self):
        from sqlalchemy.exc import IntegrityError, StatementError
        # sqlalchemy ORM
        model = self.model
        # start empty
        self.assertTrue(len(model.get_dataset()) == 0)
        self.assertTrue(len(model.get_file()) == 0)
        with self.assertRaises(KeyError):
            model.add_files(0, [File(path='')])  # can't add a file for a dataset that doesn't exist
        # add dataset
        with self.assertRaises(IntegrityError):
            model.add_dataset(Dataset(name=None, parameter=0))  # not nullable
        model.session.rollback()
        d = Dataset(name='one', parameter=0)
        self.assertTrue(d.id == None)
        model.add_dataset(d)
        self.assertTrue(d.id == 1)
        with self.assertRaises(IntegrityError):
            model.add_dataset(Dataset(name='one', parameter=0))  # must be unique
        model.session.rollback()
        model.add_dataset(Dataset(name='two', parameter=0))
        self.assertTrue(len(model.get_dataset()) == 2)
        # add files
        with self.assertRaises(IntegrityError):
            model.add_files(1, [File(path=None)])  # not nullable
        model.session.rollback()
        model.add_files(1, [File(path='1'), File(path='2')])
        model.add_files(2, [File(path='3'), File(path='4')])
        with self.assertRaises(IntegrityError):
            model.add_files(1, [File(path='4')])  # files must be unique
        model.session.rollback()
        with self.assertRaises(KeyError):
            model.add_files(3, [File(path='')])  # can't add a file for a dataset that doesn't exist
        self.assertTrue(len(model.get_file()) == 4)
        self.assertTrue(len(model.get_file(dataset_id=1)) == 2)
        self.assertTrue(len(model.get_file(dataset_id=2)) == 2)
        self.assertTrue(len(model.get_file(dataset_id=3)) == 0)
        # update via index
        f = model.get_file(3)
        f[0] = 'update'
        model.session.commit()
        f = model.get_file(3)
        self.assertTrue(f.path == 'update')
        # update errors
        d = model.get_dataset(1)
        d[0] = 'two'
        with self.assertRaises(IntegrityError):  # not unique
            model.session.commit()
        model.session.rollback()
        d[1] = 'hi'
        with self.assertRaises(StatementError):  # can't assign string to int
            model.session.commit()
        model.session.rollback()
        # delete file
        model.del_file(4)
        with self.assertRaises(KeyError):
            model.del_file(4)
        self.assertTrue(len(model.get_file()) == 3)
        # delete dataset
        model.del_dataset(1)  # this cascade deletes associated files
        self.assertTrue(len(model.get_dataset()) == 1)
        self.assertTrue(len(model.get_file()) == 1)
        with self.assertRaises(KeyError):
            model.del_dataset(1)
        # print(model.get_dataset())
        # print(model.get_file())
