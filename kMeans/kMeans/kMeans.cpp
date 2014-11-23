/*
* File Name: KMeans.cpp
* Author: lucio bale
* Date: 2014.11.23
* Description: 
	1.类内数据为欧拉距离
	2.k值人工设定
	3.初始聚类中心为数据集最前的dimen个数据
	4.聚类中心为聚类的平均值
*/
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;
// 数据对象，size为维度
struct Vector 
{
	double* coords; // 所有维度的数值
	int     size;   // 数据量
	Vector() :  coords(0), size(0) {} 
	Vector(int d) { create(d); }
	// 创建维度为d的数据，并将各维度初始化为0
	void create(int d)
	{
		size = d;
		coords = new double[size];//定义一个double型的指针，指向一块容量为size的double型变量的首地址
		for (int i=0; i<size; i++)
			coords[i] = 0.0;
	}
	// 复制一个数据
	void copy(const Vector& other)
	{
		if (size == 0) // 如果原来没有数据，创建之
			create(other.size);

		for (int i=0; i<size; i++)
			coords[i] = other.coords[i];
	}
	// 将另一个数据的各个维度加在自身的维度上
	void add(const Vector& other)
	{
		for (int i=0; i<size; i++)
			coords[i] += other.coords[i];
	}
	// 释放数值的空间
	~Vector()
	{
		if(coords)
			delete[] coords;
		size = 0;
	}
};
// 聚类结构
struct Cluster 
{
	Vector center;    // 中心的数据
	int*   member;    // 该聚类中各个数据的索引
	int    memberNum; // 数据的数量
};

// KMeans算法类
class KMeans
{
private:
	int      num;          // 输入数据的数量
	int      dimen;        // 数据的维数
	int      clusterNum;   // 数据的聚类数
	Vector*  observations; // 所有数据存放在这个数组中
	Cluster* clusters;     // 聚类数组
	int      passNum;      // 迭代的趟数
public:
	// 初始化参数和动态分配内存
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
	// 释放内存
	~KMeans()
	{
		for (int k=0; k<clusterNum; k++)
			delete [] clusters[k].member;
		delete [] clusters;
	}

	void initClusters()
	{
		// 由于初始数据中心是任意的，
		// 所以直接把前个数据作为NumClusters个聚类的数据中心
		for (int i=0; i<clusterNum; i++)
		{
			clusters[i].member[0] = i;                // 记录这个数据的索引到第i个聚类中
			clusters[i].center.copy(observations[i]); // 把这个数据作为数据中心
		}
	}
	void run()
	{
		bool converged = false; // 是否收敛
		passNum = 0;
		while (!converged && passNum < 1999)   // 如果没有收敛，则再次迭代
			// 正常情况下总是会收敛，passNum < 999是防万一
		{
			distribute();                     // 将所有数据分配到聚中心最近的聚类,为1次迭代
			converged = recalculateCenters(); // 计算新的聚类中心，如果计算结果和上次相同，认为已经收敛
			if( converged==true )
				cout << "iterations= " << passNum << endl;
			passNum++;
		}
	}
	void distribute()
	{
		// 将上次的记录的该聚类中的数据数量清0，重新开始分配数据
		for(int k=0; k<clusterNum; k++)
			getCluster(k).memberNum = 0;
		// 找出每个数据的最近聚类数据中心，并将该数据分配到该聚类
		for(int i=0; i<num; i++)
		{
			Cluster& cluster = getCluster(closestCluster(i)); // 找出最接近的其中心的聚类
			int memID = cluster.memberNum; // memberNum是当前记录的数据数量，也是新加入数据在member数组中的位置
			cluster.member[memID] = i;     // 将数据索引加入Member数组末尾，该索引为在observation数组中的位置
			cluster.memberNum++;           // 聚类中的数据数量加1
		}
	}
	//计算最近的聚类
	int closestCluster(int id)
	{
		int clusterID = 0;               // 暂时假定索引为id的数据最接近第一个聚类
		double minDist = eucNorm(id, 0); // 计算到第一个聚类中心的欧拉距离的平方
		// 计算其它聚类中心到数据的误差，找出其中最小的一个
		for (int k=1; k<clusterNum; k++) 
		{
			double d = eucNorm(id, k);
			if(d < minDist) // 如果小于前最小值，将改值作为当前最小值
			{
				minDist = d;
				clusterID = k;
			}
		}
		return clusterID;
	}
	// 索引为id的数据到第k个聚类中心的欧拉距离的平方
	double eucNorm(int id, int k)
	{
		Vector& observ = observations[id];
		Vector& center = clusters[k].center;
		double sumOfSquare = 0;
		// 将每个维度的差的平方相加，得到距离的平方
		for (int d=0; d<dimen; d++)
		{
			double dist = observ.coords[d] - center.coords[d]; // 在一个维度上中心到数据的距离
			sumOfSquare += dist*dist;
		}
		return sumOfSquare;
	}
	// 重新计算聚类中心
	bool recalculateCenters()
	{
		bool converged = true;

		for (int k=0; k<clusterNum; k++)
		{
			Cluster& cluster = getCluster(k);
			Vector average(dimen); // 初始的数据平均值
			// 统计这个聚类中数据的总和(因为在构造函数中会将各维数值清0，所以可以直接加)
			for (int m=0; m<cluster.memberNum; m++)
				average.add(observations[cluster.member[m]]);
			// 计算各个维度的评价值
			for(int d=0; d<dimen; d++)
			{
				average.coords[d] /= cluster.memberNum;
				if(average.coords[d] != cluster.center.coords[d]) // 如果和原来的聚类中心不同
					// 表示没有收敛
				{
					converged = false;
					cluster.center.coords[d] = average.coords[d]; // 用这次的平均值作为新的聚类中心
				}
			}
		}
		return converged;
	}
	// 获得第id个聚类
	Cluster& getCluster(int id)
	{
		return clusters[id];
	}
};
// 打印一个数据
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
	// 从input输入中获取数据
	int n, dimen, k;
	// 文本文件中头三个数据分别是数据数量(n)、数据维度(dimen)和聚类数量(k)
	input >> n >> dimen >> k;
	// 创建存储数据的数值
	cout << "n=" << n << ",dimen=" << dimen << ",k=" << k << endl;
	Vector* obs = new Vector[n];
	// 将数据读入数组
	for (int i=0; i<n; i++)
	{
		obs[i].create(dimen); // 创建数据
		// 依次读入各个维度的数值
		for (int d=0; d<dimen; d++)
		{
			input >> obs[i].coords[d];
		}
	}
	// 建立KMeans算法类实例
	KMeans kmeans(n, dimen, k, obs);
	kmeans.initClusters(); // 初始化
	cout << "calculating..." << endl;
	kmeans.run();          // 执行算法 

	// 输出聚类数据，如果希望输出到文件中，
	// 将后面的output的定义改为下面的形式即可
	// ofstream output("result.txt");
	ostream& output = cout;
	for (int c=0; c<k; c++)
	{
		Cluster& cluster = kmeans.getCluster(c);
		output << "---- 第" << (c + 1) << "个聚类 ----\n"; // 显示第c个聚类
		output << "聚类中心：";
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