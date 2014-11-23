/*
* File Name: KMeans.cpp
* Author: lucio bale
* Date: 2014.11.23
* Description: 
	1.��������Ϊŷ������
	2.kֵ�˹��趨
	3.��ʼ��������Ϊ���ݼ���ǰ��dimen������
	4.��������Ϊ�����ƽ��ֵ
*/
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;
// ���ݶ���sizeΪά��
struct Vector 
{
	double* coords; // ����ά�ȵ���ֵ
	int     size;   // ������
	Vector() :  coords(0), size(0) {} 
	Vector(int d) { create(d); }
	// ����ά��Ϊd�����ݣ�������ά�ȳ�ʼ��Ϊ0
	void create(int d)
	{
		size = d;
		coords = new double[size];//����һ��double�͵�ָ�룬ָ��һ������Ϊsize��double�ͱ������׵�ַ
		for (int i=0; i<size; i++)
			coords[i] = 0.0;
	}
	// ����һ������
	void copy(const Vector& other)
	{
		if (size == 0) // ���ԭ��û�����ݣ�����֮
			create(other.size);

		for (int i=0; i<size; i++)
			coords[i] = other.coords[i];
	}
	// ����һ�����ݵĸ���ά�ȼ��������ά����
	void add(const Vector& other)
	{
		for (int i=0; i<size; i++)
			coords[i] += other.coords[i];
	}
	// �ͷ���ֵ�Ŀռ�
	~Vector()
	{
		if(coords)
			delete[] coords;
		size = 0;
	}
};
// ����ṹ
struct Cluster 
{
	Vector center;    // ���ĵ�����
	int*   member;    // �þ����и������ݵ�����
	int    memberNum; // ���ݵ�����
};

// KMeans�㷨��
class KMeans
{
private:
	int      num;          // �������ݵ�����
	int      dimen;        // ���ݵ�ά��
	int      clusterNum;   // ���ݵľ�����
	Vector*  observations; // �������ݴ�������������
	Cluster* clusters;     // ��������
	int      passNum;      // ����������
public:
	// ��ʼ�������Ͷ�̬�����ڴ�
	KMeans(int n, int d, int k, Vector* ob)
		: num(n)
		, dimen(d)
		, clusterNum(k)
		, observations(ob)
		, clusters(new Cluster[k])
	{
		for (int k=0; k<clusterNum; k++)
			clusters[k].member = new int[n];
	}
	// �ͷ��ڴ�
	~KMeans()
	{
		for (int k=0; k<clusterNum; k++)
			delete [] clusters[k].member;
		delete [] clusters;
	}

	void initClusters()
	{
		// ���ڳ�ʼ��������������ģ�
		// ����ֱ�Ӱ�ǰ��������ΪNumClusters���������������
		for (int i=0; i<clusterNum; i++)
		{
			clusters[i].member[0] = i;                // ��¼������ݵ���������i��������
			clusters[i].center.copy(observations[i]); // �����������Ϊ��������
		}
	}
	void run()
	{
		bool converged = false; // �Ƿ�����
		passNum = 0;
		while (!converged && passNum < 1999)   // ���û�����������ٴε���
			// ������������ǻ�������passNum < 999�Ƿ���һ
		{
			distribute();                     // ���������ݷ��䵽����������ľ���,Ϊ1�ε���
			converged = recalculateCenters(); // �����µľ������ģ�������������ϴ���ͬ����Ϊ�Ѿ�����
			if( converged==true )
				cout << "iterations= " << passNum << endl;
			passNum++;
		}
	}
	void distribute()
	{
		// ���ϴεļ�¼�ĸþ����е�����������0�����¿�ʼ��������
		for(int k=0; k<clusterNum; k++)
			getCluster(k).memberNum = 0;
		// �ҳ�ÿ�����ݵ���������������ģ����������ݷ��䵽�þ���
		for(int i=0; i<num; i++)
		{
			Cluster& cluster = getCluster(closestCluster(i)); // �ҳ���ӽ��������ĵľ���
			int memID = cluster.memberNum; // memberNum�ǵ�ǰ��¼������������Ҳ���¼���������member�����е�λ��
			cluster.member[memID] = i;     // ��������������Member����ĩβ��������Ϊ��observation�����е�λ��
			cluster.memberNum++;           // �����е�����������1
		}
	}
	//��������ľ���
	int closestCluster(int id)
	{
		int clusterID = 0;               // ��ʱ�ٶ�����Ϊid��������ӽ���һ������
		double minDist = eucNorm(id, 0); // ���㵽��һ���������ĵ�ŷ�������ƽ��
		// ���������������ĵ����ݵ����ҳ�������С��һ��
		for (int k=1; k<clusterNum; k++) 
		{
			double d = eucNorm(id, k);
			if(d < minDist) // ���С��ǰ��Сֵ������ֵ��Ϊ��ǰ��Сֵ
			{
				minDist = d;
				clusterID = k;
			}
		}
		return clusterID;
	}
	// ����Ϊid�����ݵ���k���������ĵ�ŷ�������ƽ��
	double eucNorm(int id, int k)
	{
		Vector& observ = observations[id];
		Vector& center = clusters[k].center;
		double sumOfSquare = 0;
		// ��ÿ��ά�ȵĲ��ƽ����ӣ��õ������ƽ��
		for (int d=0; d<dimen; d++)
		{
			double dist = observ.coords[d] - center.coords[d]; // ��һ��ά�������ĵ����ݵľ���
			sumOfSquare += dist*dist;
		}
		return sumOfSquare;
	}
	// ���¼����������
	bool recalculateCenters()
	{
		bool converged = true;

		for (int k=0; k<clusterNum; k++)
		{
			Cluster& cluster = getCluster(k);
			Vector average(dimen); // ��ʼ������ƽ��ֵ
			// ͳ��������������ݵ��ܺ�(��Ϊ�ڹ��캯���лὫ��ά��ֵ��0�����Կ���ֱ�Ӽ�)
			for (int m=0; m<cluster.memberNum; m++)
				average.add(observations[cluster.member[m]]);
			// �������ά�ȵ�����ֵ
			for(int d=0; d<dimen; d++)
			{
				average.coords[d] /= cluster.memberNum;
				if(average.coords[d] != cluster.center.coords[d]) // �����ԭ���ľ������Ĳ�ͬ
					// ��ʾû������
				{
					converged = false;
					cluster.center.coords[d] = average.coords[d]; // ����ε�ƽ��ֵ��Ϊ�µľ�������
				}
			}
		}
		return converged;
	}
	// ��õ�id������
	Cluster& getCluster(int id)
	{
		return clusters[id];
	}
};
// ��ӡһ������
void printVector(ostream& output, const Vector& v)
{
	for (int i=0; i<v.size; i++)
	{
		if(i != 0)
			output << ",";
		output << v.coords[i];
	}
}
void partitionObservations(istream& input)
{
	// ��input�����л�ȡ����
	int n, dimen, k;
	// �ı��ļ���ͷ�������ݷֱ�����������(n)������ά��(dimen)�;�������(k)
	input >> n >> dimen >> k;
	// �����洢���ݵ���ֵ
	cout << "n=" << n << ",dimen=" << dimen << ",k=" << k << endl;
	Vector* obs = new Vector[n];
	// �����ݶ�������
	for (int i=0; i<n; i++)
	{
		obs[i].create(dimen); // ��������
		// ���ζ������ά�ȵ���ֵ
		for (int d=0; d<dimen; d++)
		{
			input >> obs[i].coords[d];
		}
	}
	// ����KMeans�㷨��ʵ��
	KMeans kmeans(n, dimen, k, obs);
	kmeans.initClusters(); // ��ʼ��
	cout << "calculating..." << endl;
	kmeans.run();          // ִ���㷨 

	// ����������ݣ����ϣ��������ļ��У�
	// �������output�Ķ����Ϊ�������ʽ����
	// ofstream output("result.txt");
	ostream& output = cout;
	for (int c=0; c<k; c++)
	{
		Cluster& cluster = kmeans.getCluster(c);
		output << "---- ��" << (c + 1) << "������ ----\n"; // ��ʾ��c������
		output << "�������ģ�";
		printVector(output, cluster.center);
		output << "\n" << endl;
	/*	for (int m=0; m<cluster.memberNum; m++)
		{
			int id = cluster.member[m];
			printVector(output, obs[id]);
			output << "\n";
		}
		output << endl;
	*/
	}
	delete[] obs;
}
int main()
{
	const char* fileName = "C:/Users/lenovo/Desktop/data.txt";
	ifstream obIn(fileName);
	if (obIn.is_open())
		partitionObservations(obIn);
	else
		cout << "open " << fileName << " is fail!" << endl;
	system("pause");
	return 0;
}